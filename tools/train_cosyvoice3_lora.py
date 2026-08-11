#!/usr/bin/env python3
from __future__ import print_function

import argparse
import datetime
import logging
import os
from copy import deepcopy

import deepspeed
import torch
import torch.distributed as dist
import yaml
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.scheduler import WarmupLR, WarmupPolicy, NoamHoldAnnealing, ConstantLR
from cosyvoice.utils.train_utils import (
    check_modify_and_save_config,
    init_dataset_and_dataloader,
    init_distributed,
    init_summarywriter,
    wrap_cuda_model,
)

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    try:
        from peft import TaskType
    except ImportError:  # pragma: no cover
        TaskType = None
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA training. Install with: pip install peft"
    ) from exc


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 LoRA fine-tuning (LLM only).")
    parser.add_argument("--train_engine",
                        default="torch_ddp",
                        choices=["torch_ddp", "deepspeed"],
                        help="Engine for paralleled training")
    parser.add_argument("--model", required=True, help="model which will be trained (use llm)")
    parser.add_argument("--ref_model", required=False, help="ref model used in dpo")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument("--qwen_pretrain_path", required=False, help="qwen pretrain path")
    parser.add_argument("--checkpoint", help="checkpoint model (full model weights)")
    parser.add_argument("--model_dir", required=True, help="save LoRA checkpoints to this dir")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="tensorboard log dir")
    parser.add_argument("--ddp.dist_backend",
                        dest="dist_backend",
                        default="nccl",
                        choices=["nccl", "gloo"],
                        help="distributed backend")
    parser.add_argument("--num_workers", default=0, type=int, help="num of subprocess workers for reading")
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Use pinned memory buffers")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision training")
    parser.add_argument("--dpo", action="store_true", default=False, help="Use Direct Preference Optimization")
    parser.add_argument("--timeout", default=60, type=int, help="timeout (in seconds) of cosyvoice_join")
    parser.add_argument(
        "--early-stop-on-cv-overfit",
        action="store_true",
        default=False,
        help="Stop after the first epoch where the configured CV patience is exhausted",
    )
    parser = deepspeed.add_config_arguments(parser)

    # LoRA options
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"], help="LoRA bias config")
    parser.add_argument("--lora-target-modules",
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated target module names")
    parser.add_argument("--lora-modules-to-save",
                        default="",
                        help="Comma-separated module names to keep trainable alongside LoRA")
    parser.add_argument("--lora-unfreeze",
                        default="",
                        help="Comma-separated parameter name prefixes to unfreeze")
    parser.add_argument("--lora-checkpoint",
                        default="",
                        help="Path to an existing LoRA adapter to resume from (adapter dir)")
    args = parser.parse_args()
    return args


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_by_prefix(model, prefixes: list[str]):
    if not prefixes:
        return
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = True


def apply_lora_to_cosyvoice3(model, args):
    if not hasattr(model, "llm"):
        raise RuntimeError("Expected CosyVoice3LM with .llm attribute")
    encoder = model.llm
    if not hasattr(encoder, "model"):
        raise RuntimeError("Expected Qwen2Encoder with .model attribute")
    base = encoder.model

    lora_targets = parse_list(args.lora_target_modules)
    modules_to_save = parse_list(args.lora_modules_to_save) or None

    if args.lora_checkpoint:
        peft_model = PeftModel.from_pretrained(base, args.lora_checkpoint, is_trainable=True)
    else:
        lora_kwargs = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": lora_targets,
            "bias": args.lora_bias,
            "modules_to_save": modules_to_save,
        }
        if TaskType is not None:
            lora_kwargs["task_type"] = TaskType.CAUSAL_LM
        lora_cfg = LoraConfig(**lora_kwargs)
        peft_model = get_peft_model(base, lora_cfg)
    encoder.model = peft_model

    # After PEFT wrapping, the upstream CosyVoice3LM forward path accesses
    # encoder.model.model.embed_tokens, expecting Qwen2Model. But now:
    #   encoder.model = PeftModel
    #   encoder.model.model = Qwen2ForCausalLM  (not Qwen2Model)
    #   encoder.model.model.model = Qwen2Model   (has embed_tokens)
    # Fix: proxy embed_tokens on Qwen2ForCausalLM so the upstream path works.
    qwen2_causal = peft_model.model  # Qwen2ForCausalLM
    if not hasattr(qwen2_causal, "embed_tokens") and hasattr(qwen2_causal, "model"):
        qwen2_causal.embed_tokens = qwen2_causal.model.embed_tokens

    return peft_model


def init_optimizer_and_scheduler_lora(args, configs, model, gan):
    if gan is True:
        raise RuntimeError("LoRA training script supports LLM only (gan=False).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after applying LoRA.")

    if configs["train_conf"]["optim"] == "adam":
        optimizer = torch.optim.Adam(trainable_params, **configs["train_conf"]["optim_conf"])
    elif configs["train_conf"]["optim"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, **configs["train_conf"]["optim_conf"])
    else:
        raise ValueError("unknown optimizer: " + configs["train_conf"])

    if configs["train_conf"]["scheduler"] == "warmuplr":
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs["train_conf"]["scheduler_conf"])
    elif configs["train_conf"]["scheduler"] == "warmupconstant":
        scheduler_type = WarmupPolicy
        scheduler = WarmupPolicy(optimizer, **configs["train_conf"]["scheduler_conf"])
    elif configs["train_conf"]["scheduler"] == "NoamHoldAnnealing":
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs["train_conf"]["scheduler_conf"])
    elif configs["train_conf"]["scheduler"] == "constantlr":
        scheduler_type = ConstantLR
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError("unknown scheduler: " + configs["train_conf"])

    if args.train_engine == "deepspeed":
        if scheduler_type is ConstantLR:
            def scheduler_fn(opt):
                return scheduler_type(opt)
        else:
            def scheduler_fn(opt):
                return scheduler_type(opt, **configs["train_conf"]["scheduler_conf"])
        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=None,
            lr_scheduler=scheduler_fn,
            model_parameters=trainable_params,
        )

    return model, optimizer, scheduler, None, None


def save_lora_checkpoint(model, model_name, info_dict):
    rank = int(os.environ.get("RANK", 0))
    model_dir = info_dict["model_dir"]
    if rank != 0:
        return

    os.makedirs(model_dir, exist_ok=True)
    tag_dir = os.path.join(model_dir, model_name)
    os.makedirs(tag_dir, exist_ok=True)

    base_model = unwrap_model(model)
    if not hasattr(base_model, "llm") or not hasattr(base_model.llm, "model"):
        raise RuntimeError("Expected CosyVoice3LM with Qwen2Encoder for LoRA saving")

    peft_model = base_model.llm.model
    if hasattr(peft_model, "save_pretrained"):
        peft_model.save_pretrained(tag_dir)
    else:
        raise RuntimeError("LoRA model missing save_pretrained; did you apply PEFT?")

    info_path = os.path.join(model_dir, f"{model_name}.yaml")
    info_dict = dict(info_dict)
    info_dict["save_time"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(info_path, "w", encoding="utf-8") as fout:
        yaml.dump(info_dict, fout)
    logging.info("[Rank %s] LoRA checkpoint saved to %s", rank, tag_dir)


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    if args.model != "llm":
        raise RuntimeError("LoRA script expects --model llm (CosyVoice3 LLM).")
    if args.dpo:
        raise RuntimeError("LoRA script does not support DPO training.")

    gan = False
    override_dict = {k: None for k in ["llm", "flow", "hift", "hifigan"] if k != args.model}
    with open(args.config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides={**override_dict, "qwen_pretrain_path": args.qwen_pretrain_path})
    configs["train_conf"].update(vars(args))

    init_distributed(args)
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(args, configs, gan, args.dpo)
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    # build model
    model = configs[args.model]
    start_step, start_epoch = 0, -1

    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            if "step" in state_dict:
                start_step = state_dict["step"]
            if "epoch" in state_dict:
                start_epoch = state_dict["epoch"]
        else:
            logging.warning("checkpoint %s does not exist", args.checkpoint)

    # freeze base and apply LoRA
    freeze_all_params(model)
    peft_model = apply_lora_to_cosyvoice3(model, args)
    unfreeze_by_prefix(model, parse_list(args.lora_unfreeze))

    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    # push model to device / ddp or ds
    model = wrap_cuda_model(args, model)

    # init optimizer and scheduler for LoRA params
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler_lora(args, configs, model, gan)

    # patch executor save_model to LoRA saver
    import cosyvoice.utils.executor as executor_module
    executor_module.save_model = save_lora_checkpoint

    info_dict = deepcopy(configs["train_conf"])
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    info_dict["lora"] = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "bias": args.lora_bias,
        "target_modules": parse_list(args.lora_target_modules),
        "modules_to_save": parse_list(args.lora_modules_to_save),
        "unfreeze": parse_list(args.lora_unfreeze),
        "checkpoint": args.lora_checkpoint or "",
    }

    save_lora_checkpoint(model, "init", info_dict)

    executor = Executor(gan=gan, ref_model=None, dpo_loss=None)
    executor.step = start_step

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    logging.info("start step %s start epoch %s", start_step, start_epoch)

    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)
        if args.early_stop_on_cv_overfit and int(info_dict.get("cv_overfit_flag", 0)) == 1:
            if dist.get_rank() == 0:
                logging.warning(
                    "Early stopping after epoch %s: CV %s did not improve for %s epochs.",
                    epoch,
                    info_dict.get("cv_monitor_used", info_dict.get("cv_monitor", "loss")),
                    info_dict.get("cv_no_improve_epochs", "unknown"),
                )
            break


if __name__ == "__main__":
    main()

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GuardedResumeIntegrationTests(unittest.TestCase):
    def test_cli_exposes_explicit_guarded_resume_controls(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        for flag in (
            "--guarded-checkpoints",
            "--resume-from",
            "--trust-resume-state",
            "--trust-model-checkpoint",
            "--resume-keep-last",
            "--seed",
            "--deterministic",
        ):
            self.assertIn(flag, source)
        self.assertIn("adapter-only warm start", source)

    def test_resume_validates_before_adapter_and_runtime_loads(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        main = source[source.index("def main():") :]
        validate = main.index("validate_checkpoint(")
        adapter_load = main.index("peft_model = apply_lora_to_cosyvoice3")
        runtime_load = main.index("restore_runtime_state(")
        self.assertLess(validate, adapter_load)
        self.assertLess(validate, runtime_load)

    def test_runtime_state_covers_optimizer_scheduler_scaler_and_rng(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        for token in (
            '"optimizer": canonicalize_state(optimizer.state_dict())',
            '"scheduler": canonicalize_state(scheduler.state_dict())',
            "canonicalize_state(scaler.state_dict())",
            '"python_rng": random.getstate()',
            '"numpy_rng": np.random.get_state()',
            '"torch_rng": torch.get_rng_state()',
            '"cuda_rng": torch.cuda.get_rng_state_all()',
            'random.setstate(state["python_rng"])',
            'np.random.set_state(state["numpy_rng"])',
            'torch.set_rng_state(state["torch_rng"])',
            'torch.cuda.set_rng_state_all(state["cuda_rng"])',
        ):
            self.assertIn(token, source)

    def test_runtime_evidence_canonicalizes_tensor_storage(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def canonicalize_state(value):", source)
        self.assertIn('.to(device="cpu").contiguous().clone()', source)
        self.assertIn(
            '"optimizer": canonicalize_state(optimizer.state_dict())', source
        )
        self.assertIn(
            '"scheduler": canonicalize_state(scheduler.state_dict())', source
        )

    def test_deterministic_controls_are_bound_before_training(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        main = source[source.index("def main():") :]
        configure = main.index("configure_reproducibility(args)")
        config_load = main.index("load_hyperpyyaml(")
        self.assertLess(configure, config_load)
        for token in (
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"',
            "torch.use_deterministic_algorithms(True)",
            "torch.backends.cuda.matmul.allow_tf32 = False",
            "torch.backends.cudnn.allow_tf32 = False",
            "torch.backends.cudnn.benchmark = False",
            "torch.backends.cudnn.deterministic = True",
            '"seed": args.seed',
            '"deterministic": args.deterministic',
        ):
            self.assertIn(token, source)

    def test_fresh_guarded_run_publishes_bound_initial_adapter(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        publish = source.index("initial_adapter = publish_initial_adapter(")
        contract = source.index(
            "contract = guarded_contract(args, configs, initial_adapter)", publish
        )
        wrap = source.index("model = wrap_cuda_model(args, model)")
        self.assertLess(publish, contract)
        self.assertLess(contract, wrap)

    def test_peft_config_is_normalized_for_all_checkpoint_paths(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        contract = (ROOT / "tools" / "cosyvoice_resume_contract.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("normalize_adapter_config(tag_dir)", source)
        self.assertEqual(contract.count("normalize_adapter_config(partial)"), 2)
        self.assertIn(
            'for key in ("exclude_modules", "modules_to_save", "target_modules")',
            contract,
        )

    def test_monitor_state_preserves_early_stop_history(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        for key in (
            "best_cv_epoch",
            "cv_metric",
            "cv_no_improve_epochs",
            "cv_overfit_flag",
        ):
            self.assertIn(f'"{key}"', source)
        self.assertIn('info_dict.update(resume_state.get("monitor_state", {}))', source)

    def test_epoch_checkpoint_occurs_after_early_stop_synchronization(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        sync = source.index("should_stop = synchronize_early_stop")
        publish = source.index("published = publish_checkpoint(", sync)
        branch = source.index("if should_stop:", publish)
        self.assertLess(sync, publish)
        self.assertLess(publish, branch)

    def test_single_process_boundary_is_source_enforced(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('args.train_engine != "torch_ddp"', source)
        self.assertIn("dist.get_world_size() != 1", source)
        self.assertIn("args.num_workers != 0", source)
        self.assertIn('"multi-rank state need a collective protocol"', source)

    def test_guarded_inference_exports_cannot_overwrite(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("os.path.lexists(tag_dir)", source)
        self.assertIn("Refusing to overwrite or adopt LoRA export", source)

    def test_lifecycle_opts_in_only_for_supported_topology(self) -> None:
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'os.environ["TRAIN_ENGINE"] == "torch_ddp" and settings["TRAIN_PROCESSES"] == "1"',
            source,
        )
        self.assertIn('["--guarded-checkpoints", "--trust-model-checkpoint"]', source)
        self.assertIn(
            '["--resume-from", settings["RESUME_FROM"], "--trust-resume-state"]', source
        )
        self.assertIn('"--seed",\n        settings["TRAIN_SEED"]', source)
        self.assertIn('if settings["DETERMINISTIC"] == "1":', source)

    def test_training_script_remains_valid_python(self) -> None:
        source = (ROOT / "tools" / "train_cosyvoice3_lora.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()

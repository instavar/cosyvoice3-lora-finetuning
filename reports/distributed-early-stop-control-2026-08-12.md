# Distributed early-stop control smoke

Date: 2026-08-12

## Result

The early-stop helper passed a live two-rank Gloo control test on
`desktop_tailscale`. Rank 0 supplied `local_stop=true`; rank 1 supplied
`local_stop=false`. Both ranks returned `synchronized_stop=true` after the MAX
all-reduce and reached the final barrier.

- environment: `cosyvoice`, PyTorch 2.3.1+cu121
- launcher: `python -m torch.distributed.run --standalone --nproc_per_node=2`
- ranks observed: 2
- rank-local decisions: one true, one false
- synchronized decisions: two true
- remote evidence:
  `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260812_cosyvoice_distributed_early_stop_control`

This confirms that the extracted control helper gives every initialized rank
the same stop decision when one rank requests a stop. The trainer calls this
helper before any rank breaks the epoch loop.

It does not establish convergence behavior, CV metric correctness, checkpoint
quality, or successful early stopping during a real multi-rank CosyVoice model
run. That production-path test still requires a bounded distributed training
experiment with enough VRAM and recorded CV history.

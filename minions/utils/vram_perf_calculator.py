#!/usr/bin/env python3
"""
LLM VRAM & Performance Calculator  (Inference mode)

- Inspiration:
* https://apxml.com/tools/vram-calculator
* https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm
---------------------------------------------------
Now with:
• Bigger MODEL_DB      (Llama‑3‑70B, Mixtral‑8x22B, Gemma‑7B, …)
• Bigger GPU_DB        (A6000, V100‑32GB, Apple‑M‑series, …)
• list_available_models() / list_available_gpus()
  plus CLI flags --list-models / --list-gpus
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List
import sys

# ──────────────────────────────────────────────────────────────────────────────
# 1. Reference databases
# ──────────────────────────────────────────────────────────────────────────────

# --- GPU catalogue (GiB) ---
GPU_DB: Dict[str, int] = {
    # NVIDIA ▸ consumer
    "rtx_3060_12gb": 12,
    "rtx_4090_24gb": 24,
    # NVIDIA ▸ datacenter
    "a6000_48gb": 48,
    "a100_40gb": 40,
    "h100_80gb": 80,
    "v100_32gb": 32,
    # Apple Silicon (unified memory)
    "m1_pro_16gb": 16,
    "m1_max_32gb": 32,
    "m2_pro_19gb": 19,
    "m2_max_38gb": 38,
    "m2_ultra_64gb": 64,
    # AMD Radeon RX 6000 (RDNA 2)
    "radeon_rx6800_16gb": 16,
    "radeon_rx6800xt_16gb": 16,
    "radeon_rx6900xt_16gb": 16,
    "radeon_rx6950xt_16gb": 16,
    # AMD Radeon RX 7000 (RDNA 3)
    "radeon_rx7900xt_20gb": 20,
    "radeon_rx7900xtx_24gb": 24,
    # AMD Radeon PRO (workstation)
    "radeon_pro_w6800_32gb": 32,
    "radeon_pro_w7800_32gb": 32,
    "radeon_pro_w7900_48gb": 48,
    # AMD Instinct accelerators
    "instinct_mi250x_128gb": 128,
    "instinct_mi300x_192gb": 192,
    # AMD Ryzen integrated GPUs (unified memory; typical configs)
    "ryzen_680m_16gb": 16,  # e.g. 680M iGPU, 16 GB system RAM
    "ryzen_780m_16gb": 16,  # 780M iGPU, 16 GB system RAM
}

# --- Model metadata ---
# Each entry: total parameter count, hidden‑size, layer count.
MODEL_DB: Dict[str, Dict] = {
    "deepseek-r1-3b": dict(params=3_000_000_000, hidden=2048, layers=30),
    "llama-3-8b": dict(params=8_000_000_000, hidden=4096, layers=32),
    "llama-3.2-3b": dict(params=3_000_000_000, hidden=4096, layers=32),
    "llama-3-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    "mistral-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "gemma-7b": dict(params=7_000_000_000, hidden=3072, layers=28),
    "phi-3-mini-4b": dict(params=4_000_000_000, hidden=2560, layers=32),
    "codellama-34b": dict(params=34_000_000_000, hidden=7168, layers=48),
    "mixtral-8x22b": dict(params=176_000_000_000, hidden=8192, layers=64),  # 8×22B MoE
    # add more as needed …
}

# --- Precision map (bytes/element) ---
DTYPE_SIZE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

FRAMEWORK_OVERHEAD_GB = 1.0  # buffer for CUDA / Metal / runtime arenas


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helper functionality
# ──────────────────────────────────────────────────────────────────────────────
def bytes2gib(x: int) -> float:
    return x / 1024**3


def estimate_vram(meta: Dict, dtype_bytes: float, batch: int, seq: int):
    """Return total VRAM and component breakdown (bytes)."""
    weight_b = meta["params"] * dtype_bytes
    kv_b = 2 * meta["layers"] * meta["hidden"] * seq * batch * dtype_bytes
    act_b = 0.20 * weight_b
    total_b = weight_b + kv_b + act_b + FRAMEWORK_OVERHEAD_GB * 1024**3
    return total_b, weight_b, kv_b, act_b


def list_available_models() -> List[str]:
    """Return sorted list of model keys."""
    return sorted(MODEL_DB.keys())


def list_available_gpus() -> List[str]:
    """Return sorted list of GPU keys."""
    return sorted(GPU_DB.keys())


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLI & top‑level
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Model key (see --list-models)")
    p.add_argument("--dtype", default="fp16", choices=DTYPE_SIZE.keys())
    p.add_argument("--gpu", help="GPU key or integer GiB (see --list-gpus)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument(
        "--list-models", action="store_true", help="Print available model keys and exit"
    )
    p.add_argument(
        "--list-gpus", action="store_true", help="Print available GPU keys and exit"
    )
    args = p.parse_args()

    if args.list_models:
        print("\n".join(list_available_models()))
        sys.exit(0)
    if args.list_gpus:
        print("\n".join(list_available_gpus()))
        sys.exit(0)

    # Validation —
    if args.model is None or args.gpu is None:
        p.error("--model and --gpu are required unless listing.")

    if args.model not in MODEL_DB:
        raise ValueError(
            f"Unknown model '{args.model}'. " "Use --list-models to see options."
        )

    meta = MODEL_DB[args.model]
    dtype_bytes = DTYPE_SIZE[args.dtype.lower()]

    # GPU capacity: accept int (GiB) or key
    gpu_key = args.gpu.lower()
    if gpu_key in GPU_DB:  # recognised key
        gpu_vram_gib = GPU_DB[gpu_key]
    else:  # treat as a raw GiB number
        try:
            gpu_vram_gib = float(args.gpu)
        except ValueError:
            raise ValueError(
                f"Unknown GPU '{args.gpu}'. "
                "Use --list-gpus for the supported keys or pass a number in GiB."
            )

    total_b, w_b, kv_b, act_b = estimate_vram(meta, dtype_bytes, args.batch, args.seq)

    utilisation_pct = 100 * bytes2gib(total_b) / gpu_vram_gib
    rating = (
        "LOW"
        if utilisation_pct < 60
        else (
            "MODERATE"
            if utilisation_pct < 80
            else "HIGH" if utilisation_pct < 95 else "CRITICAL"
        )
    )

    # Report —
    print(f"\n=== VRAM estimate for {args.model} ({args.dtype.upper()}) ===")
    print(f"Batch {args.batch} | Sequence {args.seq} tokens")
    print(
        f"Total: {bytes2gib(total_b):.2f} GiB  "
        f"({utilisation_pct:.1f}% of {gpu_vram_gib} GiB) — {rating}"
    )
    print("Breakdown:")
    print(f"  • Model weights : {bytes2gib(w_b):.2f} GiB")
    print(f"  • KV cache      : {bytes2gib(kv_b):.2f} GiB")
    print(f"  • Activations   : {bytes2gib(act_b):.2f} GiB")
    print(f"  • Overhead      : {FRAMEWORK_OVERHEAD_GB:.2f} GiB\n")


if __name__ == "__main__":
    main()

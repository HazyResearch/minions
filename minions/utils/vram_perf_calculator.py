#!/usr/bin/env python3
"""
LLM VRAM & Performance Calculator  (Inference mode)

- Inspiration:
* https://apxml.com/tools/vram-calculator
* https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm
---------------------------------------------------
Now with:
• Expanded MODEL_DB (Llama 4, Qwen 3, OLMo 3, Nemotron 3, DeepSeek V3/R1, etc.)
• Expanded GPU_DB (M3/M4 Apple Silicon, RTX 50 series, etc.)
• list_available_models() / list_available_gpus()
  plus CLI flags --list-models / --list-gpus
• Support for MoE models (showing both total and active parameters)
"""

import argparse
import subprocess
from typing import Dict, List, Optional
import sys

# ──────────────────────────────────────────────────────────────────────────────
# 1. Reference databases
# ──────────────────────────────────────────────────────────────────────────────

# --- GPU catalogue (GiB) ---
GPU_DB: Dict[str, int] = {
    # NVIDIA ▸ consumer RTX 40 series
    "rtx_4060_8gb": 8,
    "rtx_4060ti_8gb": 8,
    "rtx_4060ti_16gb": 16,
    "rtx_4070_12gb": 12,
    "rtx_4070s_12gb": 12,
    "rtx_4070ti_12gb": 12,
    "rtx_4070tis_16gb": 16,
    "rtx_4080_16gb": 16,
    "rtx_4080s_16gb": 16,
    "rtx_4090_24gb": 24,
    # NVIDIA ▸ consumer RTX 50 series (Blackwell, 2025)
    "rtx_5070_12gb": 12,
    "rtx_5070ti_16gb": 16,
    "rtx_5080_16gb": 16,
    "rtx_5090_32gb": 32,
    # NVIDIA ▸ consumer RTX 30 series (legacy)
    "rtx_3060_12gb": 12,
    "rtx_3070_8gb": 8,
    "rtx_3080_10gb": 10,
    "rtx_3080_12gb": 12,
    "rtx_3090_24gb": 24,
    "rtx_3090ti_24gb": 24,
    # NVIDIA ▸ datacenter / professional
    "a6000_48gb": 48,
    "a100_40gb": 40,
    "a100_80gb": 80,
    "h100_80gb": 80,
    "h200_141gb": 141,
    "l40s_48gb": 48,
    "v100_16gb": 16,
    "v100_32gb": 32,
    "rtx_pro_6000_48gb": 48,
    # Apple Silicon M1 series (unified memory)
    "m1_8gb": 8,
    "m1_16gb": 16,
    "m1_pro_16gb": 16,
    "m1_pro_32gb": 32,
    "m1_max_32gb": 32,
    "m1_max_64gb": 64,
    "m1_ultra_64gb": 64,
    "m1_ultra_128gb": 128,
    # Apple Silicon M2 series (unified memory)
    "m2_8gb": 8,
    "m2_16gb": 16,
    "m2_24gb": 24,
    "m2_pro_16gb": 16,
    "m2_pro_32gb": 32,
    "m2_max_32gb": 32,
    "m2_max_64gb": 64,
    "m2_max_96gb": 96,
    "m2_ultra_64gb": 64,
    "m2_ultra_128gb": 128,
    "m2_ultra_192gb": 192,
    # Apple Silicon M3 series (unified memory)
    "m3_8gb": 8,
    "m3_16gb": 16,
    "m3_24gb": 24,
    "m3_pro_18gb": 18,
    "m3_pro_36gb": 36,
    "m3_max_36gb": 36,
    "m3_max_48gb": 48,
    "m3_max_64gb": 64,
    "m3_max_96gb": 96,
    "m3_max_128gb": 128,
    "m3_ultra_96gb": 96,
    "m3_ultra_256gb": 256,
    "m3_ultra_512gb": 512,
    # Apple Silicon M4 series (unified memory)
    "m4_16gb": 16,
    "m4_24gb": 24,
    "m4_32gb": 32,
    "m4_pro_24gb": 24,
    "m4_pro_48gb": 48,
    "m4_pro_64gb": 64,
    "m4_max_36gb": 36,
    "m4_max_48gb": 48,
    "m4_max_64gb": 64,
    "m4_max_128gb": 128,
    # AMD Radeon RX 6000 (RDNA 2)
    "radeon_rx6800_16gb": 16,
    "radeon_rx6800xt_16gb": 16,
    "radeon_rx6900xt_16gb": 16,
    "radeon_rx6950xt_16gb": 16,
    # AMD Radeon RX 7000 (RDNA 3)
    "radeon_rx7600_8gb": 8,
    "radeon_rx7700xt_12gb": 12,
    "radeon_rx7800xt_16gb": 16,
    "radeon_rx7900gre_16gb": 16,
    "radeon_rx7900xt_20gb": 20,
    "radeon_rx7900xtx_24gb": 24,
    # AMD Radeon RX 9000 (RDNA 4)
    "radeon_rx9070_16gb": 16,
    "radeon_rx9070xt_16gb": 16,
    # AMD Radeon PRO (workstation)
    "radeon_pro_w6800_32gb": 32,
    "radeon_pro_w7800_32gb": 32,
    "radeon_pro_w7900_48gb": 48,
    # AMD Instinct accelerators
    "instinct_mi250x_128gb": 128,
    "instinct_mi300x_192gb": 192,
    # AMD Ryzen integrated GPUs (unified memory; typical configs)
    "ryzen_680m_16gb": 16,
    "ryzen_780m_16gb": 16,
}

# --- Model metadata ---
# Each entry: total parameter count, hidden-size, layer count
# For MoE models: params = total params, active_params = params active per token
MODEL_DB: Dict[str, Dict] = {
    # ─── DeepSeek models ───
    "deepseek-r1-1.5b": dict(params=1_500_000_000, hidden=1536, layers=28),
    "deepseek-r1-7b": dict(params=7_000_000_000, hidden=3584, layers=28),
    "deepseek-r1-8b": dict(params=8_000_000_000, hidden=4096, layers=32),
    "deepseek-r1-14b": dict(params=14_000_000_000, hidden=5120, layers=48),
    "deepseek-r1-32b": dict(params=32_000_000_000, hidden=5120, layers=64),
    "deepseek-r1-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    "deepseek-v3-671b": dict(params=671_000_000_000, active_params=37_000_000_000, hidden=7168, layers=61, is_moe=True),
    "deepseek-r1-671b": dict(params=671_000_000_000, active_params=37_000_000_000, hidden=7168, layers=61, is_moe=True),
    # ─── Llama 3 / 3.1 / 3.2 / 3.3 models ───
    "llama-3-8b": dict(params=8_000_000_000, hidden=4096, layers=32),
    "llama-3-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    "llama-3.1-8b": dict(params=8_000_000_000, hidden=4096, layers=32),
    "llama-3.1-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    "llama-3.1-405b": dict(params=405_000_000_000, hidden=16384, layers=126),
    "llama-3.2-1b": dict(params=1_000_000_000, hidden=2048, layers=16),
    "llama-3.2-3b": dict(params=3_000_000_000, hidden=3072, layers=28),
    "llama-3.3-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    # ─── Llama 4 models (MoE) ───
    "llama-4-scout-109b": dict(params=109_000_000_000, active_params=17_000_000_000, hidden=5120, layers=48, is_moe=True),
    "llama-4-maverick-400b": dict(params=400_000_000_000, active_params=17_000_000_000, hidden=12288, layers=120, is_moe=True),
    # ─── Mistral models ───
    "mistral-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "mistral-nemo-12b": dict(params=12_000_000_000, hidden=5120, layers=40),
    "mistral-small-24b": dict(params=24_000_000_000, hidden=6144, layers=56),
    "mixtral-8x7b": dict(params=46_700_000_000, active_params=12_900_000_000, hidden=4096, layers=32, is_moe=True),
    "mixtral-8x22b": dict(params=176_000_000_000, active_params=39_000_000_000, hidden=6144, layers=56, is_moe=True),
    # ─── Qwen 2.5 models ───
    "qwen2.5-0.5b": dict(params=500_000_000, hidden=896, layers=24),
    "qwen2.5-1.5b": dict(params=1_500_000_000, hidden=1536, layers=28),
    "qwen2.5-3b": dict(params=3_000_000_000, hidden=2048, layers=36),
    "qwen2.5-7b": dict(params=7_000_000_000, hidden=3584, layers=28),
    "qwen2.5-14b": dict(params=14_000_000_000, hidden=5120, layers=48),
    "qwen2.5-32b": dict(params=32_000_000_000, hidden=5120, layers=64),
    "qwen2.5-72b": dict(params=72_000_000_000, hidden=8192, layers=80),
    # ─── Qwen 3 models (dense) ───
    "qwen3-0.6b": dict(params=600_000_000, hidden=1024, layers=28),
    "qwen3-1.7b": dict(params=1_700_000_000, hidden=2048, layers=28),
    "qwen3-4b": dict(params=4_000_000_000, hidden=2560, layers=36),
    "qwen3-8b": dict(params=8_000_000_000, hidden=4096, layers=36),
    "qwen3-14b": dict(params=14_000_000_000, hidden=5120, layers=40),
    "qwen3-32b": dict(params=32_000_000_000, hidden=5120, layers=64),
    # ─── Qwen 3 models (MoE) ───
    "qwen3-30b-a3b": dict(params=30_000_000_000, active_params=3_000_000_000, hidden=2048, layers=48, is_moe=True),
    "qwen3-235b-a22b": dict(params=235_000_000_000, active_params=22_000_000_000, hidden=4096, layers=94, is_moe=True),
    "qwen3-coder-480b-a35b": dict(params=480_000_000_000, active_params=35_000_000_000, hidden=5120, layers=94, is_moe=True),
    # ─── NVIDIA Nemotron 3 models (MoE with Mamba-2) ───
    "nemotron3-nano-30b-a3b": dict(params=30_000_000_000, active_params=3_500_000_000, hidden=4096, layers=29, is_moe=True),
    "nemotron3-super-100b-a10b": dict(params=100_000_000_000, active_params=10_000_000_000, hidden=5120, layers=48, is_moe=True),
    "nemotron3-ultra-500b-a50b": dict(params=500_000_000_000, active_params=50_000_000_000, hidden=8192, layers=80, is_moe=True),
    # ─── OLMo 3 models ───
    "olmo3-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "olmo3-32b": dict(params=32_000_000_000, hidden=5120, layers=64),
    # ─── Gemma models ───
    "gemma-2b": dict(params=2_000_000_000, hidden=2048, layers=18),
    "gemma-7b": dict(params=7_000_000_000, hidden=3072, layers=28),
    "gemma2-2b": dict(params=2_000_000_000, hidden=2304, layers=26),
    "gemma2-9b": dict(params=9_000_000_000, hidden=3584, layers=42),
    "gemma2-27b": dict(params=27_000_000_000, hidden=4608, layers=46),
    "gemma3-1b": dict(params=1_000_000_000, hidden=1536, layers=26),
    "gemma3-4b": dict(params=4_000_000_000, hidden=2560, layers=34),
    "gemma3-12b": dict(params=12_000_000_000, hidden=3840, layers=48),
    "gemma3-27b": dict(params=27_000_000_000, hidden=5120, layers=62),
    # ─── Phi models ───
    "phi-3-mini-4b": dict(params=4_000_000_000, hidden=3072, layers=32),
    "phi-3-small-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "phi-3-medium-14b": dict(params=14_000_000_000, hidden=5120, layers=40),
    "phi-4-14b": dict(params=14_000_000_000, hidden=5120, layers=40),
    # ─── Code Llama models ───
    "codellama-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "codellama-13b": dict(params=13_000_000_000, hidden=5120, layers=40),
    "codellama-34b": dict(params=34_000_000_000, hidden=8192, layers=48),
    "codellama-70b": dict(params=70_000_000_000, hidden=8192, layers=80),
    # ─── StarCoder models ───
    "starcoder2-3b": dict(params=3_000_000_000, hidden=2560, layers=30),
    "starcoder2-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "starcoder2-15b": dict(params=15_000_000_000, hidden=6144, layers=40),
    # ─── Command-R models ───
    "command-r-35b": dict(params=35_000_000_000, hidden=8192, layers=40),
    "command-r-plus-104b": dict(params=104_000_000_000, hidden=12288, layers=64),
    # ─── Yi models ───
    "yi-1.5-6b": dict(params=6_000_000_000, hidden=4096, layers=32),
    "yi-1.5-9b": dict(params=9_000_000_000, hidden=4096, layers=48),
    "yi-1.5-34b": dict(params=34_000_000_000, hidden=7168, layers=60),
    # ─── Falcon models ───
    "falcon-7b": dict(params=7_000_000_000, hidden=4544, layers=32),
    "falcon-40b": dict(params=40_000_000_000, hidden=8192, layers=60),
    "falcon-180b": dict(params=180_000_000_000, hidden=14848, layers=80),
    # ─── Other notable models ───
    "internlm2-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "internlm2-20b": dict(params=20_000_000_000, hidden=5120, layers=48),
    "baichuan2-7b": dict(params=7_000_000_000, hidden=4096, layers=32),
    "baichuan2-13b": dict(params=13_000_000_000, hidden=5120, layers=40),
}

# --- Precision map (bytes/element) ---
DTYPE_SIZE = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp8": 1.0,
    "fp4": 0.5,
}

FRAMEWORK_OVERHEAD_GB = 1.0  # buffer for CUDA / Metal / runtime arenas


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helper functionality
# ──────────────────────────────────────────────────────────────────────────────
def bytes2gib(x: float) -> float:
    return x / (1024**3)


def estimate_vram(meta: Dict, dtype_bytes: float, batch: int, seq: int) -> tuple:
    """Return total VRAM and component breakdown (bytes).
    
    For MoE models, we use active_params for weight memory estimation
    since only active experts are loaded per forward pass in optimized inference.
    For KV cache, we still need to account for attention layers.
    """
    # For MoE models, use active params for weight estimation
    effective_params = meta.get("active_params", meta["params"])
    
    weight_b = effective_params * dtype_bytes
    
    # KV cache: 2 (K+V) * layers * hidden * seq * batch * dtype
    # Note: For MoE models, attention is typically shared, so we use full layers
    kv_b = 2 * meta["layers"] * meta["hidden"] * seq * batch * dtype_bytes
    
    # Activations estimate (roughly 20% of weights for inference)
    act_b = 0.20 * weight_b
    
    total_b = weight_b + kv_b + act_b + FRAMEWORK_OVERHEAD_GB * (1024**3)
    
    return total_b, weight_b, kv_b, act_b


def list_available_models() -> List[str]:
    """Return sorted list of model keys."""
    return sorted(MODEL_DB.keys())


def list_available_gpus() -> List[str]:
    """Return sorted list of GPU keys."""
    return sorted(GPU_DB.keys())


def format_params(params: int) -> str:
    """Format parameter count in human-readable form."""
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    elif params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    else:
        return f"{params:,}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. HuggingFace Hub Model Memory Estimation (via hf-mem)
#    https://github.com/alvarobartt/hf-mem
# ──────────────────────────────────────────────────────────────────────────────

def run_hf_mem(model_id: str, revision: str = "main") -> Optional[str]:
    """
    Run hf-mem CLI to estimate inference memory requirements for a HuggingFace model.
    
    Uses the hf-mem package (https://github.com/alvarobartt/hf-mem) to fetch
    Safetensors metadata from the HuggingFace Hub and estimate memory requirements.
    
    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-7B")
        revision: Git revision/branch (default: "main")
    
    Returns:
        Output from hf-mem CLI as string, or None if failed.
    
    Example:
        >>> output = run_hf_mem("meta-llama/Llama-3.2-1B")
        >>> print(output)
    
    Note:
        Requires hf-mem to be installed. Install with: pip install hf-mem
        Or run directly with uvx: uvx hf-mem --model-id <model_id>
    """
    import os
    
    # Build command
    cmd = ["uvx", "hf-mem", "--model-id", model_id]
    if revision != "main":
        cmd.extend(["--revision", revision])
    
    # Add HF_TOKEN if available
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            # Try with pip-installed hf-mem
            cmd[0] = "hf-mem"
            cmd.remove("hf-mem")  # Remove duplicate
            cmd = ["hf-mem", "--model-id", model_id]
            if revision != "main":
                cmd.extend(["--revision", revision])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"Error running hf-mem: {result.stderr}", file=sys.stderr)
                return None
                
    except FileNotFoundError:
        print("Error: hf-mem not found. Install with: pip install hf-mem", file=sys.stderr)
        print("Or run directly with: uvx hf-mem --model-id <model_id>", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("Error: hf-mem timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running hf-mem: {e}", file=sys.stderr)
        return None


def estimate_hf_model_memory(model_id: str, revision: str = "main") -> None:
    """
    Estimate and print inference memory requirements for a HuggingFace model.
    
    Uses hf-mem (https://github.com/alvarobartt/hf-mem) to fetch Safetensors 
    metadata via HTTP Range requests and estimate memory requirements without 
    downloading the full model.
    
    Works with Transformers, Diffusers, and Sentence Transformers models.
    
    Args:
        model_id: HuggingFace model ID (e.g., "MiniMaxAI/MiniMax-M2", "Qwen/Qwen2.5-7B")
        revision: Git revision/branch (default: "main")
    
    Example:
        >>> estimate_hf_model_memory("meta-llama/Llama-3.2-1B")
        >>> estimate_hf_model_memory("Qwen/Qwen-Image")  # Diffusers model
    
    Note:
        Requires hf-mem: pip install hf-mem (or use uvx hf-mem)
    """
    print(f"Fetching model metadata from HuggingFace Hub: {model_id}...")
    output = run_hf_mem(model_id, revision)
    
    if output:
        print(output)
    else:
        print(f"\nFailed to estimate memory for {model_id}")
        print("Make sure hf-mem is installed: pip install hf-mem")
        print(f"Or run directly: uvx hf-mem --model-id {model_id}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. CLI & top‑level
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="LLM VRAM Calculator - Estimate GPU memory requirements for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model llama-3-8b --dtype fp16 --gpu rtx_4090_24gb
  %(prog)s --model qwen3-32b --dtype int4 --gpu m4_max_64gb --seq 4096
  %(prog)s --model deepseek-v3-671b --dtype fp8 --gpu h100_80gb --batch 4
  %(prog)s --hf-model meta-llama/Llama-3.2-1B --dtype int4 --gpu rtx_4090_24gb
  %(prog)s --hf-model Qwen/Qwen2.5-7B --dtype fp16
  %(prog)s --list-models
  %(prog)s --list-gpus
        """
    )
    p.add_argument("--model", help="Model key from built-in database (see --list-models)")
    p.add_argument("--hf-model", dest="hf_model", 
                   help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B). "
                        "Fetches metadata directly from HuggingFace Hub.")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPE_SIZE.keys()),
                   help="Data type/precision (default: fp16)")
    p.add_argument("--gpu", help="GPU key or integer GiB (see --list-gpus)")
    p.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--seq", type=int, default=2048, help="Sequence length in tokens (default: 2048)")
    p.add_argument("--revision", default="main", help="HuggingFace model revision (default: main)")
    p.add_argument(
        "--list-models", action="store_true", help="Print available model keys and exit"
    )
    p.add_argument(
        "--list-gpus", action="store_true", help="Print available GPU keys and exit"
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Show additional model details"
    )
    args = p.parse_args()

    if args.list_models:
        print("Available models:")
        print("-" * 60)
        for model in list_available_models():
            meta = MODEL_DB[model]
            is_moe = meta.get("is_moe", False)
            if is_moe:
                total = format_params(meta["params"])
                active = format_params(meta["active_params"])
                print(f"  {model:<35} {total} total ({active} active) [MoE]")
            else:
                params = format_params(meta["params"])
                print(f"  {model:<35} {params}")
        sys.exit(0)
        
    if args.list_gpus:
        print("Available GPUs:")
        print("-" * 50)
        
        # Group by category
        categories = {
            "NVIDIA RTX 50": [],
            "NVIDIA RTX 40": [],
            "NVIDIA RTX 30": [],
            "NVIDIA Datacenter": [],
            "Apple M4": [],
            "Apple M3": [],
            "Apple M2": [],
            "Apple M1": [],
            "AMD Radeon RX": [],
            "AMD Radeon PRO": [],
            "AMD Instinct": [],
            "AMD Ryzen": [],
        }
        
        for gpu in list_available_gpus():
            vram = GPU_DB[gpu]
            entry = f"  {gpu:<30} {vram:>3} GiB"
            
            if gpu.startswith("rtx_5"):
                categories["NVIDIA RTX 50"].append(entry)
            elif gpu.startswith("rtx_4"):
                categories["NVIDIA RTX 40"].append(entry)
            elif gpu.startswith("rtx_3"):
                categories["NVIDIA RTX 30"].append(entry)
            elif any(x in gpu for x in ["a100", "a6000", "h100", "h200", "l40", "v100", "pro_6000"]):
                categories["NVIDIA Datacenter"].append(entry)
            elif gpu.startswith("m4"):
                categories["Apple M4"].append(entry)
            elif gpu.startswith("m3"):
                categories["Apple M3"].append(entry)
            elif gpu.startswith("m2"):
                categories["Apple M2"].append(entry)
            elif gpu.startswith("m1"):
                categories["Apple M1"].append(entry)
            elif gpu.startswith("radeon_rx"):
                categories["AMD Radeon RX"].append(entry)
            elif gpu.startswith("radeon_pro"):
                categories["AMD Radeon PRO"].append(entry)
            elif gpu.startswith("instinct"):
                categories["AMD Instinct"].append(entry)
            elif gpu.startswith("ryzen"):
                categories["AMD Ryzen"].append(entry)
        
        for cat, gpus in categories.items():
            if gpus:
                print(f"\n{cat}:")
                for g in gpus:
                    print(g)
        
        sys.exit(0)

    # Handle HuggingFace model fetching via hf-mem
    if args.hf_model:
        estimate_hf_model_memory(args.hf_model, args.revision)
        sys.exit(0)

    # Validation for built-in models —
    if args.model is None or args.gpu is None:
        p.error("--model (or --hf-model) and --gpu are required unless listing.")

    if args.model not in MODEL_DB:
        print(f"Error: Unknown model '{args.model}'.", file=sys.stderr)
        print("Use --list-models to see available options, or use --hf-model to fetch from HuggingFace Hub.", file=sys.stderr)
        sys.exit(1)

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
            print(f"Error: Unknown GPU '{args.gpu}'.", file=sys.stderr)
            print("Use --list-gpus for supported keys or pass a number in GiB.", file=sys.stderr)
            sys.exit(1)

    total_b, w_b, kv_b, act_b = estimate_vram(meta, dtype_bytes, args.batch, args.seq)
    total_gib = bytes2gib(total_b)

    utilisation_pct = 100 * total_gib / gpu_vram_gib
    
    if utilisation_pct < 60:
        rating = "LOW"
        rating_color = "\033[92m"  # green
    elif utilisation_pct < 80:
        rating = "MODERATE"
        rating_color = "\033[93m"  # yellow
    elif utilisation_pct < 95:
        rating = "HIGH"
        rating_color = "\033[91m"  # red
    elif utilisation_pct <= 100:
        rating = "CRITICAL"
        rating_color = "\033[91m"  # red
    else:
        rating = "EXCEEDS VRAM"
        rating_color = "\033[91m"  # red
    
    reset_color = "\033[0m"

    # Report —
    is_moe = meta.get("is_moe", False)
    
    print(f"\n{'='*60}")
    print(f"VRAM ESTIMATE: {args.model} ({args.dtype.upper()})")
    print(f"{'='*60}")
    
    if is_moe:
        total_params = format_params(meta["params"])
        active_params = format_params(meta["active_params"])
        print(f"Model Type    : MoE ({total_params} total, {active_params} active)")
    else:
        print(f"Model Type    : Dense ({format_params(meta['params'])})")
    
    print(f"Batch Size    : {args.batch}")
    print(f"Sequence Len  : {args.seq:,} tokens")
    print(f"GPU Target    : {gpu_vram_gib} GiB")
    print()
    print(f"Total VRAM    : {total_gib:.2f} GiB")
    print(f"Utilization   : {rating_color}{utilisation_pct:.1f}% — {rating}{reset_color}")
    print()
    print("Breakdown:")
    print(f"  • Model weights : {bytes2gib(w_b):.2f} GiB")
    print(f"  • KV cache      : {bytes2gib(kv_b):.2f} GiB")
    print(f"  • Activations   : {bytes2gib(act_b):.2f} GiB")
    print(f"  • Overhead      : {FRAMEWORK_OVERHEAD_GB:.2f} GiB")
    
    if args.verbose:
        print()
        print("Model Architecture:")
        print(f"  • Hidden size   : {meta['hidden']:,}")
        print(f"  • Layers        : {meta['layers']}")
        if is_moe:
            print(f"  • Architecture  : Mixture of Experts")
    
    if utilisation_pct > 100:
        print()
        print(f"\033[91mWarning: Model exceeds available VRAM by {total_gib - gpu_vram_gib:.2f} GiB")
        print(f"Consider using a smaller dtype (e.g., int4) or a GPU with more memory.\033[0m")
    
    print()


if __name__ == "__main__":
    main()
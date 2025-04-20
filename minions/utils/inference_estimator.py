#!/usr/bin/env python3
"""
inference_estimator.py  ·  v0.3
===============================

*   Peak‑compute model:       same as v0.2 (FP16/INT8 tensor‑core lookup)
*   Memory‑bandwidth model:   BW / (bytes‑per‑token) roofline
*   Empirical calibration:    one‑shot timing → factor cached at ~/.cache/ie_calib.json
"""

from __future__ import annotations
import argparse, json, os, subprocess, time
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch


# --------------------------------------------------------------------------- #
#  tiny helpers
# --------------------------------------------------------------------------- #


def _run(cmd: str, timeout: float = 1.0) -> str | None:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, timeout=timeout
        ).strip()
    except subprocess.SubprocessError:
        return None


def _cache_path() -> Path:
    path = Path.home() / ".cache" / "ie_calib.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}")
    return path


# --------------------------------------------------------------------------- #
#  Hardware profiling
# --------------------------------------------------------------------------- #


@dataclass
class HardwareProfiler:
    # flags
    has_gpu: bool = False
    has_mps: bool = False
    has_cpu: bool = False

    # compute
    num_devices: int = 0
    cores_per_dev: int = 0
    clock_hz: float = 0.0
    flops_per_core_clk: float = 0.0  # FMAs⇒FLOPs

    # bandwidth
    mem_bw_Bps: float = 0.0  # bytes / second

    # ------------------------------------------------------------------ #

    @classmethod
    def profile(cls) -> "HardwareProfiler":
        p = cls()

        # ---------- CUDA ------------------------------------------------ #
        if torch.cuda.is_available():
            p.has_gpu = True
            props = torch.cuda.get_device_properties(0)
            p.num_devices = torch.cuda.device_count()
            p.cores_per_dev = props.multi_processor_count

            # tensor‑core FLOPs per SM per clk
            sm_flops = {(8, 0): 256, (8, 6): 256, (9, 0): 512, (9, 8): 1024}
            p.flops_per_core_clk = sm_flops.get((props.major, props.minor), 128 * 2)

            # core clock
            clk = _run("nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader")
            try:
                p.clock_hz = float(clk.split()[0]) * 1e6
            except (AttributeError, ValueError):
                p.clock_hz = 1.5e9

            # memory bandwidth (2× because GDDR/GDDR6X = DDR)
            mem_clk = props.memory_clock_rate * 1e3  # Hz
            bus_bits = props.memory_bus_width
            p.mem_bw_Bps = 2 * mem_clk * bus_bits / 8  # B/s

        # ---------- Apple‑Silicon MPS ----------------------------------- #
        elif torch.backends.mps.is_available():
            p.has_mps = True
            p.num_devices = 1
            p.cores_per_dev = (
                int(
                    (
                        _run(
                            "system_profiler SPDisplaysDataType | grep 'Total Number of Cores' | head -1"
                        )
                        or "0"
                    ).split(":")[-1]
                )
                or 16
            )
            p.clock_hz = 1.45e9
            p.flops_per_core_clk = 512  # FP16
            p.mem_bw_Bps = 300e9  # ≈M3‑Max DRAM BW

        # ---------- CPU fallback ---------------------------------------- #
        else:
            p.has_cpu = True
            p.num_devices = 1
            p.cores_per_dev = psutil.cpu_count(logical=False) or 1
            freq = psutil.cpu_freq()
            p.clock_hz = (freq.max if freq else 3000) * 1e6
            p.flops_per_core_clk = 16  # AVX‑512 FP32
            p.mem_bw_Bps = 50e9  # guesstimate

        return p

    # ------------------------------------------------------------------ #

    @property
    def peak_tflops(self) -> float:
        return (
            self.num_devices
            * self.cores_per_dev
            * self.clock_hz
            * self.flops_per_core_clk
            / 1e12
        )

    @property
    def peak_mem_GBps(self) -> float:
        return self.mem_bw_Bps / 1e9


# --------------------------------------------------------------------------- #
#  Model profiling
# --------------------------------------------------------------------------- #


@dataclass
class ModelProfiler:
    model_name: str
    num_params: int
    is_quant: bool = False
    quant_bits: int = 32

    model_to_params = {
        # … (same table as before) …
        "llama3.2": 3_000_000_000,
        "llama3.1:8b": 8_000_000_000,
        "llama3.2:1b": 1_000_000_000,
        "mistral7b": 7_000_000_000,
    }

    @classmethod
    def profile(
        cls, name: str, is_quant: bool = None, quant_bits: int = None
    ) -> "ModelProfiler":
        """
        Profile a model, with optional manual quantization parameters.

        Args:
            name: The model name from the model_to_params table
            is_quant: Manually override quantization detection
            quant_bits: Manually specify quantization bit-width (4, 8, etc.)

        Returns:
            ModelProfiler instance with the specified parameters
        """
        if name not in cls.model_to_params:
            raise KeyError(f"unknown model '{name}'")

        # Auto-detect quantization from name if not manually specified
        auto_is_quant, auto_qbits = ("bit" in name), 32
        if auto_is_quant:
            for b in (4, 8):  # crude parse
                if f"{b}bit" in name:
                    auto_qbits = b

        # Use manually specified values if provided, otherwise use auto-detected
        final_is_quant = is_quant if is_quant is not None else auto_is_quant
        final_qbits = quant_bits if quant_bits is not None else auto_qbits

        return cls(name, cls.model_to_params[name], final_is_quant, final_qbits)

    # ------------------------------------------------------------------ #
    #  compute & bandwidth requirement per *token*
    # ------------------------------------------------------------------ #

    @property
    def flops_per_tok_T(self) -> float:
        spd = 32 / self.quant_bits if self.is_quant else 1
        return 2 * self.num_params / 1e12 / spd  # TFLOPs

    @property
    def bytes_per_tok(self) -> float:
        # weight‑only bandwidth: params × bytes/weight
        wbytes = self.quant_bits / 8
        return self.num_params * wbytes  # B


# --------------------------------------------------------------------------- #
#  Inference estimator (+ empirical calibration)
# --------------------------------------------------------------------------- #


class InferenceEstimator:
    def __init__(self, model_name: str, is_quant: bool = True, quant_bits: int = 4):
        """
        Initialize the inference estimator.

        Args:
            model_name: Name of the model from the model_to_params table
            is_quant: Manually specify if the model is quantized
            quant_bits: Manually specify the quantization bit-width
        """
        self.hw = HardwareProfiler.profile()
        self.model = ModelProfiler.profile(model_name, is_quant, quant_bits)
        self._calib = self._load_calib()
        print(f"Calibration factor: {self._calib}")

    # ------------------------------ core math ------------------------- #

    def _theoretical_tok_s(self) -> float:
        comp_tok_s = self.hw.peak_tflops / self.model.flops_per_tok_T
        bw_tok_s = self.hw.mem_bw_Bps / self.model.bytes_per_tok
        return min(comp_tok_s, bw_tok_s)

    # ------------------------------ public API ------------------------ #

    def estimate(self, n_tokens: int) -> tuple[float, float]:
        tps = self._calib * self._theoretical_tok_s()
        eta = n_tokens / tps
        return tps, eta

    # ------------------------------------------------------------------ #
    #  Empirical calibration helpers
    # ------------------------------------------------------------------ #

    def calibrate(self, model_client, sample_tokens: int = 32, prompt: str = "Hello"):
        """
        Run a single timed generation to compute a correction factor and cache it.
        `model_client` must expose `.generate(prompt, max_tokens=...)`
        and return a string or list with *sample_tokens* new tokens.
        """
        print("Calibrating inference estimator...")
        # warm‑up (GPU driver lazy init)
        messages = [{"role": "user", "content": prompt}]
        model_client.chat(messages)

        torch.cuda.synchronize() if self.hw.has_gpu else None
        t0 = time.time()
        model_client.chat(messages)
        torch.cuda.synchronize() if self.hw.has_gpu else None
        dt = time.time() - t0
        meas_tps = sample_tokens / dt

        theo_tps = self._theoretical_tok_s()
        factor = meas_tps / theo_tps
        self._calib = factor
        self._save_calib()
        return factor, meas_tps

    # --------------------------- cache I/O ---------------------------- #

    def _cache_key(self) -> str:
        hw_id = "gpu" if self.hw.has_gpu else "mps" if self.hw.has_mps else "cpu"
        quant_suffix = f":{self.model.quant_bits}bit" if self.model.is_quant else ""
        return f"{hw_id}:{self.model.model_name}{quant_suffix}"

    def _load_calib(self) -> float:
        try:
            data = json.loads(_cache_path().read_text())
            return float(data.get(self._cache_key(), 1.0))
        except (OSError, ValueError):
            return 1.0

    def _save_calib(self):
        path = _cache_path()
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        data[self._cache_key()] = self._calib
        path.write_text(json.dumps(data))

    # --------------------------- pretty‑print ------------------------- #

    def describe(self, n: int) -> str:
        tps, eta = self.estimate(n)
        quant_status = (
            f"{self.model.quant_bits}-bit" if self.model.is_quant else "unquantized"
        )
        return (
            f"== Hardware =================================================\n"
            f" Peak compute : {self.hw.peak_tflops:7.1f} TFLOPs\n"
            f" Peak BW      : {self.hw.peak_mem_GBps:7.1f} GB/s\n"
            f"== Model (#tok={n}) =========================================\n"
            f" Quantization : {quant_status}\n"
            f" Cost/token   : {self.model.flops_per_tok_T:7.3f} TFLOPs\n"
            f" Bytes/token  : {self.model.bytes_per_tok/1e6:7.1f} MB\n"
            f"== Throughput ===============================================\n"
            f" Theory (un‑cal.): {self._theoretical_tok_s():8.1f} tok/s\n"
            f" Empirical factor : {self._calib:8.3f}×\n"
            f" Estimate          {tps:8.1f} tok/s   ETA ≈ {eta:6.2f} s"
        )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM throughput estimator with roofline + calibration"
    )
    parser.add_argument("--model", help="name from param‑table")
    parser.add_argument("--tokens", type=int, help="# input tokens")
    parser.add_argument("--describe", action="store_true", help="pretty print")
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="manually specify that model is quantized",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=[4, 8, 16],
        help="quantization bit-width (4, 8, or 16)",
    )
    args = parser.parse_args()

    # For a store_true action, it will be False if not specified
    is_quant = True if args.quantized else None
    quant_bits = args.quant_bits if args.quant_bits is not None else None

    est = InferenceEstimator(args.model, is_quant, quant_bits)
    if args.describe:
        print(est.describe(args.tokens))
    else:
        tps, eta = est.estimate(args.tokens)
        print(f"{tps:.1f} tok/s   ETA {eta:.2f}s")

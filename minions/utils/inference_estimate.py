import psutil 
import torch
import subprocess
import time

"""
In this module, we estimate the inference throughput and end-to-end latency
for a given model and input (approximate number of input tokens). To do this, 
we use the following approach: 

1. HardwareProfiler: Gathers key hardware information in order to estimate the 
peak FLOPs that can be performed by the hardware. We support (nvidia) GPUs, CPUs, 
and Apple Silicon MPS. 
2. ModelProfiler: Gathers key model information in order to estimate the peak FLOPs
required to do an inference pass.
3. InferenceProfiler: Estimates the actual throughput and latency of the model on 
the given hardware. We compute the theoretical tok/s (peak hw FLOPs / peak model FLOPs)
and ETA (number of input tokens / theoretical tok/s).
"""


class HardwareProfiler: 
    has_gpu: bool = False
    has_cpu: bool = False
    has_mps_backend: bool = False 

    _gpu_num: int = 0 
    _gpu_num_cores_per: int = 0 
    _gpu_clock_rate: float = 0
    _gpu_peak_flops_per_core: float = 0

    _cpu_num: int = 0
    _cpu_num_cores_per: int = 0 
    _cpu_clock_rate: float = 0
    _cpu_peak_flops_per_core: float = 0

    _mps_num: int = 0 
    _mps_num_cores_per: int = 0 
    _mps_clock_rate: float = 0
    _mps_peak_flops_per_core: float = 0

    @classmethod 
    def profile(cls) -> 'HardwareProfiler': 
        """
        Profile the hardware and return a HardwareProfiler instance with accurate specs.
        Detects CPU, CUDA GPUs, and Apple Silicon MPS capabilities.
        """
        profiler = cls()

        if torch.cuda.is_available(): 
            profiler.has_gpu = True 
            profiler._gpu_num = torch.cuda.device_count()

            # TODO: There are 128 CUDA cores on recent GPUs. Make this more modular.
            profiler._gpu_num_cores_per = torch.cuda.get_device_properties(0).multi_processor_count * 128
            
            try:
                result = subprocess.check_output("nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader", shell=True, text=True)
                profiler._gpu_clock_rate = float(result.split()[0].strip()) * 1e6
            except (subprocess.CalledProcessError, ValueError):
                profiler._gpu_clock_rate = 0  

            # TODO: Assumes 2 FLOPs for each FMA core. Account for tensor cores.
            profiler._gpu_peak_flops_per_core = 2

        elif torch.backends.mps.is_available(): 
            profiler.has_mps_backend = True 
            profiler._mps_num = torch.mps.device_count()

            try:
                out = subprocess.check_output("system_profiler SPDisplaysDataType | grep 'Total Number of Cores'", shell=True, text=True)
                profiler._mps_num_cores_per = int(out.split(':')[-1].strip())
            except (subprocess.CalledProcessError, ValueError):
                profiler._mps_num_cores_per = 0 

            # TODO: assumes 2 FLOPs for each FMA core and ~1500 MHz clock rate.
            profiler._mps_clock_rate = 1500 * 1e6
            profiler._mps_peak_flops_per_core = 2

        else:  
            # TODO: check if this works on apple metal?

            profiler.has_cpu_backend = True 
            profiler._cpu_num = psutil.cpu_count(logical=False)
            profiler._cpu_num_cores_per = psutil.cpu_count(logical=False)
            profiler._cpu_clock_rate = psutil.cpu_freq().max * 1e6

        return profiler


class ModelProfiler: 
    model_name: str 
    num_parameters: int 
    is_quantized: bool 
    quantization_bits: int

    # Maintain this manually
    model_to_param_count = {
        "mlx-community/Llama-3.2-3B-Instruct-4bit": 3000000000,
        "mlx-community/Qwen2.5-7B-8bit": 7000000000,
        "mlx-community/Qwen2.5-3B-8bit": 3000000000,
        "mlx-community/Llama-3.2-3B-Instruct-8bit": 3000000000,
        "mlx-community/Llama-3.1-8B-Instruct": 8000000000,
        "cartesia-ai/Llamba-8B-8bit-mlx": 8000000000,
        "cartesia-ai/Llamba-1B-4bit-mlx": 1000000000,
        "cartesia-ai/Llamba-3B-4bit-mlx": 3000000000,
        "llama3.2": 3000000000,
        "llama3.1:8b": 8000000000,
        "llama3.2:1b": 1000000000,
        "gemma3:4b": 4000000000,
        "granite3.2-vision": 3000000000,
        "phi4": 1600000000,
        "qwen2.5:1.5b": 1500000000,
        "qwen2.5:3b": 3000000000,
        "qwen2.5:7b": 7000000000,
        "qwen2.5:14b": 14000000000,
        "mistral7b": 7000000000,
        "deepseek-r1:1.5b": 1500000000,
        "deepseek-r1:7b": 7000000000,
        "deepseek-r1:8b": 8000000000,
    }

    @classmethod 
    def profile(cls, model_name: str) -> 'ModelProfiler': 
        """
        Profile a model given its name. 
        """
        profiler = cls() 

        profiler.model_name = model_name 
        profiler.num_parameters = cls.model_to_param_count[model_name]
        profiler.is_quantized = "bit" in model_name
        if profiler.is_quantized: 
            profiler.quantization_bits = int(model_name.split("bit")[0][-1]) 

        return profiler


class InferenceEstimator: 
    def __init__(self, model_client: any): 
        self.model_client = model_client
        self.hw_profile = HardwareProfiler.profile() 
        self.model_profile = ModelProfiler.profile(model_client.model_name)

    def _compute_theoretical_throughput(self, num_input_tokens: int) -> float: 
        """
            To find the theoretical throughput, we compute the 
            ratio of the peak hardware flops and peak model flops. 
        """
        def compute_hw_peak_flops() -> float: 
            device = "_mps_" if self.hw_profile.has_mps_backend else "_gpu_" if self.hw_profile.has_gpu else "_cpu_"
            return getattr(self.hw_profile, f"{device}num") * \
                   getattr(self.hw_profile, f"{device}num_cores_per") * \
                   getattr(self.hw_profile, f"{device}clock_rate") * \
                   getattr(self.hw_profile, f"{device}peak_flops_per_core")

        def compute_model_peak_flops() -> float: 
            # TODO: Assume 32-bit operations by default and "perfect" quantization speed-up 
            quantization_speed_up = 1
            if self.model_profile.is_quantized: 
                quantization_speed_up = self.model_profile.quantization_bits / 32
            
            # 2*N*D
            return 2 * self.model_profile.num_parameters * num_input_tokens * quantization_speed_up

        # compute ratio
        hw_flops = compute_hw_peak_flops()
        model_flops = compute_model_peak_flops()
        return hw_flops / model_flops 

    def estimate(self, num_input_tokens: int) -> tuple[float, float]: 
        """
            Estimate the tokens/sec and ETA. 
        """
        theoretical_throughput = self._compute_theoretical_throughput(num_input_tokens)
        eta = num_input_tokens / theoretical_throughput
        return theoretical_throughput, eta
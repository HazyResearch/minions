import psutil 
import torch
import subprocess
import time

"""
Breakdown: 
1) Hardware Profiler 
2) Model Profiler 
3) Inference Estimator  (with progress tracking)
"""


class HardwareProfiler: 
    has_gpu: bool = False
    has_cpu: bool = False
    has_mps_backend: bool = False 

    _gpu_num: int = 0 
    _gpu_num_cores_per: int = 0 
    _gpu_clock_rate: float = 0
    _gpu_peak_flops_per_core: float = 0
    # _gpu_mem_gb_total: float 

    _cpu_num: int = 0
    _cpu_num_cores_per: int = 0 
    _cpu_clock_rate: float = 0
    _cpu_peak_flops_per_core: float = 0
    # _cpu_mem_gb_total: float 

    _mps_num: int = 0 
    _mps_num_cores_per: int = 0 
    _mps_clock_rate: float = 0
    _mps_peak_flops_per_core: float = 0
    # _mps_mem_gb_total: float 

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
            profiler._gpu_num_cores_per = torch.cuda.get_device_properties(0).multi_processor_count * 128
            
            
            try:
                out = subprocess.check_output("nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader", shell=True, text=True)
                profiler._gpu_clock_rate = int(out.split()[0].strip()) / 1000  # (GHz)
            except (subprocess.CalledProcessError, ValueError):
                profiler._gpu_clock_rate = 0  

            profiler._gpu_peak_flops_per_core = 2  # (assumes 2 flops for each FMA core)

        elif torch.backends.mps.is_available(): 
            profiler.has_mps_backend = True 
            profiler._mps_num = torch.mps.device_count()

            try:
                out = subprocess.check_output("system_profiler SPDisplaysDataType | grep 'Total Number of Cores'", shell=True, text=True)
                profiler._mps_num_cores_per = int(out.split(':')[-1].strip())
            except (subprocess.CalledProcessError, ValueError):
                profiler._mps_num_cores_per = 0 

            profiler._mps_clock_rate = 1.5  # assumed frequency for mps (GHz)
            profiler._mps_peak_flops_per_core = 2  # (assumes 2 flops for each core)

        else:  
            # TODO: check if this works on apple metal?

            profiler.has_cpu_backend = True 
            profiler._cpu_num = psutil.cpu_count(logical=False)
            profiler._cpu_num_cores_per = psutil.cpu_count(logical=False)
            profiler._cpu_clock_rate = psutil.cpu_freq()

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
        profiler = cls() 

        profiler.model_name = model_name 
        profiler.num_parameters = cls.model_to_param_count[model_name]
        profiler.is_quantized = "bit" in model_name 
        profiler.quantization_bits = int(model_name.split("bit")[0][-1]) 

        return profiler


class InferenceEstimator: 
    def __init__(self, model_client: any): 
        self.model_client = model_client
        self.hw_profile = HardwareProfiler.profile() 
        self.model_profile = ModelProfiler.profile(model_client.model_name)

        self.empirical_throughput = self._compute_empirical_throughput() 

    # def _compute_empirical_throughput(self, num_trials: int = 5) -> float:
    #     """
    #         To compute the empirical throughput, we run the model on a 
    #         small batch of inputs and time how long it takes. We then 
    #         divide the total token count by the time taken to run the 
    #         model.
    #     """ 
    #     test_input = [
    #         {
    #             "role": "user", 
    #             "content": "Write a paragraph about the movie: Minions."
    #         }
    #     ] 

    #     trial_times = [] 
    #     trial_output_tok_count = [] 
    #     for _ in range(num_trials): 
    #         if self.hw_profile.has_gpu: 
    #             start = torch.cuda.Event(enable_timing=True)
    #             end = torch.cuda.Event(enable_timing=True)
    #             start.record()
    #         else: 
    #             start = time.time() 
            
    #         # Run inference 
    #         _, usage, _ = self.model_client.chat(messages=test_input)

    #         if self.hw_profile.has_gpu: 
    #             end.record()
    #             torch.cuda.synchronize()
    #             elapsed = start.elapsed_time() / 1000
    #         else: 
    #             elapsed = time.time() - start
            
    #         trial_times.append(elapsed) 
    #         trial_output_tok_count.append(usage.prompt_tokens + usage.completion_tokens)
        
    #     tokens_per_second = [output_tok_count / time for output_tok_count, time in zip(trial_output_tok_count, trial_times)] 
    #     return sum(tokens_per_second) / len(tokens_per_second)


    def _compute_theoretical_throughput(self, num_input_tokens: int) -> float: 
        """
            To find the theoretical throughput, we compute the 
            ratio of the peak hardware flops and peak model flops. 
        """
        def compute_hw_peak_flops() -> float: 
            device = "_cpu_"
            if self.hw_profile.has_gpu: 
                device = "_gpu_"
            if self.hw_profile.has_mps_backend: 
                device = "_mps_"
            
            return getattr(self.hw_profile, f"{device}num") * \
                   getattr(self.hw_profile, f"{device}num_cores_per") * \
                   getattr(self.hw_profile, f"{device}clock_rate") * 1e9 * \
                   getattr(self.hw_profile, f"{device}peak_flops_per_core")

        def compute_model_peak_flops() -> float: 
            # TODO: Assume 32-bit operations by default and "perfect" quantization speed-up 
            quantization_speed_up = 1
            if self.model_profile.is_quantized: 
                quantization_speed_up = 32 / self.model_profile.quantization_bits 
            
            # 6*N*D
            return 6 * self.model_profile.num_parameters * num_input_tokens * quantization_speed_up

        # compute ratio
        hw_flops = compute_hw_peak_flops()
        model_flops = compute_model_peak_flops()
        print(f"HW flops: {hw_flops}")
        print(f"Model flops: {model_flops}")
        return hw_flops / model_flops 

    def estimate(self, num_input_tokens: int) -> tuple[float, float]: 
        """
            We utilize the following approach: 

            T_theoretical, T_empirical, and efficiency are computed. 

            T_adjusted = T_theoretical * efficiency
            ETA = T_adjusted / num_input_tokens
            return T_adjusted, ETA 
        """
        theoretical_throughput = self._compute_theoretical_throughput(num_input_tokens)
        return theoretical_throughput, num_input_tokens / theoretical_throughput


        # efficiency_factor = theoretical_throughput / self.empirical_throughput

        # print(f"Empirical tok/s: {self.empirical_throughput}")
        # print(f"Theoretical tok/s: {theoretical_throughput}")

        # adjusted_tokens_per_second = theoretical_throughput * efficiency_factor 
        # return adjusted_tokens_per_second, adjusted_tokens_per_second / num_input_tokens 
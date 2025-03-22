from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Any
import yaml
import platform
import subprocess
import os
import argparse
import sys
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from estimate_memory import calculate_gpu_memory, DeviceType
from model_config import ModelConfig

#TODO: add QLoRA
class TrainingMethod(Enum):
    FULL_FINETUNE = "full_trn"
    LORA = "lora_trn"

@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    num_layers: int
    heads: int
    vocab_size: int
    intermediate_size: int
    
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access"""
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        """Enable 'in' operator"""
        return hasattr(self, key)

#TODO: add logic for MoE memory
class ModelArchitecture(Enum):
    DECODER_0_5B = "decoder_0.5b"
    DECODER_1B = "decoder_1b"
    DECODER_3B = "decoder_3b"
    DECODER_7B = "decoder_7b"
    DECODER_13B = "decoder_13b"
    # DECODER_34B = "decoder_34b"
    # DECODER_70B = "decoder_70b"
    # MOE_7B = "moe_7b"      # For mixture of experts models like Mixtral
    
    def get_size_billions(self) -> float:
        """Get model size in billions of parameters."""
        size_map = {
            "decoder_0.5b": 0.5,
            "decoder_1b": 1.0,
            "decoder_3b": 3.0,
            "decoder_7b": 7.0,
            "decoder_13b": 13.0,
            # "decoder_34b": 34.0, #There is no llama 30b? There is only codellama-34b
            "decoder_70b": 70.0,
            # "moe_7b": 7.0,  # Base size, actual compute for moe is different and more complicated than normal decoders
        }
        return size_map[self.value]
    
    def get_available_models(self) -> List[str]:
        """Get list of available HuggingFace model names for this architecture size."""
        model_map = {
            "decoder_0.5b": ["Qwen/Qwen2-0.5B"],
            "decoder_1b": ["meta-llama/Llama-3.2-1B"],  # Note: This is actually 1.8B but closest to 1B category
            "decoder_3b": ["meta-llama/Llama-3.2-3B"],
            "decoder_7b": ["meta-llama/Llama-2-7b-hf", "Qwen/Qwen-7B"],
            "decoder_13b": ["meta-llama/Llama-2-13b-hf", "Qwen/Qwen-14B"],  # Note: Qwen-14B is closest to 13B category
            "decoder_70b": ["meta-llama/Llama-3.1-70B", "Qwen/Qwen-72B"],
        }
        return model_map[self.value]
    
    def get_default_config(self) -> ModelConfig:
        """Get default architecture configuration."""
        configs = {
            "decoder_0.5b": ModelConfig(
                name="decoder_0.5b",
                hidden_dim=896,
                num_layers=24,
                heads=14,
                vocab_size=151936,
                intermediate_size=4864
            ),
            "decoder_1b": ModelConfig(
                name="decoder_1b",
                hidden_dim=2048,
                num_layers=16,
                heads=32,
                vocab_size=128256,
                intermediate_size=8192
            ),
            "decoder_3b": ModelConfig(
                name="decoder_3b",
                hidden_dim=3072,
                num_layers=28,
                heads=24,
                vocab_size=128256,
                intermediate_size=8192
            ),
            "decoder_7b": ModelConfig(
                name="decoder_7b",
                hidden_dim=4096,
                num_layers=32,
                heads=32,
                vocab_size=32000,   
                intermediate_size=11008
            ),
            "decoder_13b": ModelConfig(
                name="decoder_13b",
                hidden_dim=5120,
                num_layers=40,
                heads=40,
                vocab_size=32000,
                intermediate_size=13824
            ),
            "decoder_70b": ModelConfig(
                name="decoder_70b",
                hidden_dim=8192,
                num_layers=80,
                intermediate_size=28672,
                vocab_size=32000, #Higher for llama 3.1 versions
                heads=64,
            ),
        }
        return configs[self.value]

@dataclass
class HardwareSpec:
    memory: float  # in GB
    num_devices: int
    name: str
    device_type: DeviceType  # New field for device type

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training configuration based on hardware")
    
    # Hardware override options
    parser.add_argument("--manual-hardware", action="store_true",
                       help="Manually specify hardware instead of auto-detection")
    parser.add_argument("--device-type", type=str, choices=['gpu', 'cpu', 'mps'],
                       help="Type of device (required if manual-hardware is set)")
    parser.add_argument("--memory", type=float,
                       help="Available memory in GB (required if manual-hardware is set)")
    parser.add_argument("--num-devices", type=int, default=1,
                       help="Number of devices (default: 1)")
    parser.add_argument("--device-name", type=str,
                       help="Name of the device (optional)")
    
    return parser.parse_args()

def detect_hardware() -> Optional[HardwareSpec]:
    """Auto-detect hardware specifications. Returns None if detection fails."""
    # Check for NVIDIA GPU using PyTorch
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)  # Using first GPU
            memory = props.total_memory / (1024**3)  # Convert bytes to GB
            return HardwareSpec(memory=memory, num_devices=device_count, name=device_name, device_type=DeviceType.GPU)
        except Exception as e:
            print(f"Warning: Error in PyTorch GPU detection: {e}")
            print("Please use --manual-hardware to specify your GPU details")
            return None
    
    # Check for Apple Silicon
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        try:
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            mem_bytes = int(result.stdout.split(':')[1].strip())
            memory = mem_bytes / (1024**3)  # Convert to GB
            return HardwareSpec(memory=memory, num_devices=1, name="Apple Silicon", device_type=DeviceType.MLX)
        except Exception as e:
            print(f"Warning: Error detecting Apple Silicon specs: {e}")
            print("Please use --manual-hardware to specify your MPS device details")
            return None
    
    # CPU Only - Try to get allocated memory first
    try:
        # Try to get SLURM memory limit first
        slurm_mem = os.getenv('SLURM_MEM_PER_NODE')
        if slurm_mem:
            memory = float(slurm_mem) / 1024  # Convert MB to GB
            return HardwareSpec(memory=memory, num_devices=1, name="CPU (SLURM)", device_type=DeviceType.CPU)
        
        # If no SLURM, try cgroups
        if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
                memory = int(f.read()) / (1024**3)  # Convert bytes to GB
                if memory < 1000:  # Sanity check
                    return HardwareSpec(memory=memory, num_devices=1, name="CPU (cgroup)", device_type=DeviceType.CPU)
        
        print("Could not detect CPU memory limits.")
        print("Please use --manual-hardware to specify your CPU memory")
        return None
        
    except Exception as e:
        print(f"Warning: Error detecting system memory: {e}")
        print("Please use --manual-hardware to specify your system details")
        return None

class TrainingConfigurator:
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.memory_calculator = self._initialize_memory_calculator()

    def _initialize_memory_calculator(self):
        """Initialize the memory calculation logic from your existing code"""
        return calculate_gpu_memory

    def _estimate_memory_requirement(
        self, 
        model: ModelArchitecture,
        training_method: TrainingMethod,
        batch_size: int = 1,
        sequence_length: int = 2048
    ) -> float:
        """Estimate memory requirement for a given configuration"""
        
        config = model.get_default_config()
        
        memory_info = self.memory_calculator(
            config=config,
            context_len=sequence_length,
            batch_size=batch_size,
            training_type=training_method.value,
            optimizer="adam_opt",
            platform=self.hardware.device_type  # Pass the device type
        )
        
        return memory_info["Total"] / 1024

    
    #TODO: add model parallelism here (currently it is data parallelism)
    #TODO: add quantization (for now f16 training is assumed)
    #TODO: you can kind of make batch size irrelevant by using gradient accumulation. So for now lets have batch size as 1
    #TODO: Assumes no 
    def find_optimal_configuration(
        self, 
        sequence_length: int = 2048,
        min_batch_size: int = 1
    ) -> Tuple[ModelArchitecture, TrainingMethod, int]:
        """Find the largest model and best training method that fits in memory"""
        
        
        available_memory = self.hardware.memory * 1 #self.hardware.num_devices is not considered since this is data parallelism. If its fdsp we can change this

        best_config = None
        max_memory_usage = 0

        #Filter out models that are too big to even fit in memoey at f16 precision
        viable_models = []
        for model in ModelArchitecture:
            model_size = model.get_size_billions()
            if model_size <= (self.hardware.memory / 2):  # rough heuristic for f16
                viable_models.append(model)
                
        for model in viable_models:
            for training_method in TrainingMethod:
                
                memory_needed = self._estimate_memory_requirement(
                    model=model,
                    training_method=training_method,
                    batch_size=min_batch_size,
                    sequence_length=sequence_length
                )

                print(memory_needed, available_memory, training_method, model)

                if memory_needed <= available_memory and memory_needed > max_memory_usage:
                    max_memory_usage = memory_needed
                    best_config = (model, training_method, min_batch_size)

        return best_config

    def generate_deepspeed_config(
        self, 
        model: ModelArchitecture, 
        training_method: TrainingMethod,
        batch_size: int
    ) -> dict:
        """Generate DeepSpeed configuration based on selected model and training method"""
        
        # Get the default model name for this architecture
        model_name = model.get_available_models()[0]  # Use first available model as default
        
        base_config = {
            "train_batch_size": batch_size * self.hardware.num_devices,
            "gradient_accumulation_steps": 1,
            "model_parameters": {
                "model_name_or_path": model_name
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 2e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5,
                    "warmup_num_steps": 100
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                }
            }
        }

        #TODO: add qlora config (deepspeed yaml doesn't support qlora directly)
        
        if training_method == TrainingMethod.LORA:
            base_config["lora"] = {
                "target_modules": ["q_proj", "v_proj"],
                "r": 8,
                "alpha": 16,
                "dropout": 0.05
            }

        return base_config

def generate_training_config(hardware_spec: Optional[HardwareSpec] = None) -> str:
    """Main function to generate training configuration"""
    
    if hardware_spec is None:
        hardware_spec = detect_hardware()
        if hardware_spec is None:
            raise ValueError(
                "Hardware detection failed. Please use --manual-hardware with:\n"
                "  --device-type [gpu|cpu|mps]\n"
                "  --memory <memory_in_gb>\n"
                "  --num-devices <number_of_devices>\n"
                "  --device-name <optional_name>"
            )
        
        print(f"\nDetected Hardware:")
        print(f"Device: {hardware_spec.name}")
        print(f"Number of devices: {hardware_spec.num_devices}")
        print(f"Available Memory: {hardware_spec.memory:.2f} GB")
    
    configurator = TrainingConfigurator(hardware_spec)
    
    # Find optimal configuration
    model, training_method, batch_size = configurator.find_optimal_configuration(sequence_length=512)
    
    if not model:
        raise ValueError("No viable configuration found for the given hardware")
    
    # Generate DeepSpeed config
    ds_config = configurator.generate_deepspeed_config(model, training_method, batch_size)
    
    # Convert to YAML
    return yaml.dump(ds_config, default_flow_style=False)

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    
    if args.manual_hardware:
        if not args.device_type or not args.memory:
            print("Error: --device-type and --memory are required with --manual-hardware")
            print("Example: --manual-hardware --device-type gpu --memory 40 --num-devices 1")
            sys.exit(1)
            
        hardware = HardwareSpec(
            memory=args.memory,
            num_devices=args.num_devices,
            name=args.device_name or f"{args.device_type.upper()} (Manual)",
            device_type=DeviceType(args.device_type)
        )
    else:
        hardware = None  # Will trigger auto-detection
    
    try:
        yaml_config = generate_training_config(hardware)
        print(f"\nGenerated DeepSpeed configuration:\n{yaml_config}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
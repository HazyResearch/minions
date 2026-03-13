from typing import Dict, Optional, Literal, Any, Protocol
import math
from enum import Enum

#Logic ported to python from https://github.com/RahulSChand/gpu_poor/blob/main/src/App.js

class DeviceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    MLX = "mlx"

class ConfigLike(Protocol):
    def __getitem__(self, key: str) -> Any: ...

# Constants
BILLION = 1_000_000_000
TERA = 1_000_000_000 * 1_000

TrainingType = Literal["full_trn", "lora_trn"]
OptimizerType = Literal["adam_opt", "sgd_opt"]
# QuantType = Literal["no_quant", "bnb_int8", "bnb_q4"]

QuantType = Literal["float16", "float8", "float4"]

def compute_model_size(config: ConfigLike) -> float:
    """Calculate the base model size in parameters."""
    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    intermediate_size = config["intermediate_size"]
    
    return (vocab_size * hidden_dim * 2 + 
            num_layers * 4 * hidden_dim * hidden_dim + 
            num_layers * 3 * intermediate_size * hidden_dim)

def compute_activation_memory(context_len: int, config: ConfigLike, platform: DeviceType, gradient_checkpointing: bool) -> float:
    """Calculate inference-only activation memory in MB.
    
    This accounts for:
    - Attention computations (QKV, attention matrices, output projection)
    - FFN activations
    - Layer norms
    - Residual connections
    - FP32 conversions where needed
    - Doesn't take into account quanitzation sicne acitvations are usually converted to fp16
    """
    hidden_dim = config["hidden_dim"]
    heads = config["heads"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_layers"]
    float_bytes = 2  # float16 by default
    
    # Per layer calculations
    # Attention block
    attn_per_layer = (
        context_len * hidden_dim * 3 * float_bytes +  # QKV projections
        context_len * hidden_dim * 2 * float_bytes +  # QK transpose
        context_len * context_len * heads * float_bytes +  # attention matrix
        context_len * context_len * heads * 4 +  # FP32 conversion # Taken from llama's huggingface implementation
        context_len * context_len * heads * float_bytes +  # scaled attention
        context_len * hidden_dim * float_bytes +  # output projection
        context_len * hidden_dim * float_bytes +  # residual
        context_len * hidden_dim * float_bytes    # layer norm
    )
    
    # FFN block
    ffn_per_layer = (
        hidden_dim * context_len * float_bytes +  # first FFN
        hidden_dim * context_len * float_bytes +  # residual
        float_bytes * 5 * context_len * intermediate_size +  # intermediate
        intermediate_size * context_len * float_bytes  # final FFN
    )
    
    # Layer norms
    norm = context_len * 4 * 2 + context_len * hidden_dim * float_bytes * 6
    
    # Total per layer
    total_per_layer = attn_per_layer + ffn_per_layer + norm
    
    # Total across all layers
    total = total_per_layer * num_layers
    
    if platform == DeviceType.CPU:
        # CPU doesn't have the same optimizations as GPU/MLX
        # But can still use quantization via libraries like llama.cpp
        total *= 1.0  # No special optimization multiplier
    elif platform == DeviceType.MLX:
        # TODO: These optimization numbers need to be validated for MLX
        # Currently using GPU-like numbers but this needs benchmarking
        if gradient_checkpointing:
            total *= 0.15
    else:  # GPU
        if gradient_checkpointing:
            total *= 0.15
        
    
    # Convert to MB
    return total / (1024 * 1024)

# This is extra mmeory required by bitsandbytes for quantization. This is bnb specific, will need to be updated to work for other quant training methods
def get_extra_memory(config: ConfigLike, quant: str, context_len: int) -> float:
    """Calculate extra memory needed for quantization."""
    constant_8_extra = 0.75
    constant_4_extra = 1.0
    constant_qlora = 0.75
    
    common = ((10 * config["hidden_dim"] + 
              5 * config["hidden_dim"] + 
              4 * config["intermediate_size"] + 
              2 * config["intermediate_size"]) * 
             config["num_layers"])
    
    base_len = 50
    ratio_context_len = context_len / 50
    context_len_sqrt_root = math.sqrt(ratio_context_len) if ratio_context_len > 1.0 else 1.0
    
    if quant == "float8":
        return constant_8_extra * common * base_len * context_len_sqrt_root * 1.25
    elif quant == "float4":
        return constant_4_extra * common * base_len * context_len_sqrt_root
    elif quant == "qlora":
        return constant_qlora * common * base_len * context_len_sqrt_root
    return 0

def get_grad_opt_memory(
    training_type: TrainingType,
    optimizer: OptimizerType,
    quant_type: QuantType,
    model_size: float,
    config: ConfigLike,
    context_len: int,
    batch_size: int = 1
) -> float:
    """
    Calculate gradient and optimizer memory requirements.
    
    Args:
        training_type: Type of training (full, LoRA, QLoRA)
        optimizer: Optimizer type
        quant_type: Quantization type
        model_size: Model size in parameters
        config: Model configuration
        context_len: Context length
        batch_size: Batch size
    
    Returns:
        Memory requirement in bytes
    """
    # QLoRA specific calculation
    if training_type == "qlora":
        if optimizer == "adam_opt":
            memory = (config["num_layers"] * 8 * config["hidden_dim"] * 0.5 * 4 * 3 +
                     get_extra_memory(config, "qlora", context_len) * batch_size)
        else:  # sgd
            memory = (config["num_layers"] * 8 * config["hidden_dim"] * 0.5 * 4 * 1 +
                     get_extra_memory(config, "qlora", context_len) * batch_size)
        return memory

    # LoRA specific calculation
    if training_type == "lora_trn":
        if optimizer == "adam_opt":
            if quant_type == "float16":
                return config["num_layers"] * 8 * config["hidden_dim"] * 2 * 4 * 3 * 2
            else:
                return (config["num_layers"] * 8 * config["hidden_dim"] * 2 * 4 * 3 +
                       get_extra_memory(config, quant_type, context_len) * batch_size)
        else:
            if quant_type == "float16":
                return config["num_layers"] * 8 * config["hidden_dim"] * 2 * 4 * 2
            else:
                return (config["num_layers"] * 8 * config["hidden_dim"] * 2 * 4 * 1 +
                       get_extra_memory(config, quant_type, context_len) * batch_size)

    # Full training calculation
    float_bytes = 2  # float16
    if quant_type == "float8":
        float_bytes = 1
    elif quant_type == "float4":
        float_bytes = 0.5

    if optimizer == "adam_opt":
        memory = model_size * 3 * float_bytes #3 copies of the param. One for grad, two for Adam optimizer mean/var
    else:  # sgd
        memory = model_size * float_bytes

    if quant_type != "float16":
        memory += get_extra_memory(config, quant_type, context_len) * batch_size

    return memory

def calculate_gpu_memory(
    config: ConfigLike,
    context_len: int,
    batch_size: int = 1,
    quant_type: QuantType = "float16",
    training_type: Optional[TrainingType] = None,
    gradient_checkpointing: bool = False,
    optimizer: OptimizerType = "adam_opt",
    platform: DeviceType = DeviceType.GPU,  # Add platform parameter with GPU default,
) -> Dict[str, float]:
    """
    Calculate total GPU memory requirements in MB.
    
    Args:
        config: Model configuration dictionary
        context_len: Context length
        batch_size: Batch size
        quant_type: Quantization type
        training_type: Type of training (None for inference)
        gradient_checkpointing: Whether using gradient checkpointing
        optimizer: Optimizer type (for training)
        platform: Device type
    """
    # Base model size calculation
    model_size_params = compute_model_size(config)
    
    # Convert to MB and adjust for quantization
    # We can pass the model size directly from train_minions.py file as well. This is a more precise calculation of number of parameters (and therefore size)
    
    if quant_type == "float16":
        model_size_mb = model_size_params * 2.0 / (1024 * 1024)
    elif quant_type == "bfloat16":
        model_size_mb = model_size_params * 2.0 / (1024 * 1024)
    elif quant_type == "float8":
        model_size_mb = (model_size_params / 2.0) / (1024 * 1024)
    elif quant_type == "float4":
        model_size_mb = (model_size_params * 4.0) / (1024 * 1024)
    
    # Activation memory
    activation_memory = compute_activation_memory(context_len, config, platform, gradient_checkpointing)
    
    # Extra memory for quantization
    extra_memory = get_extra_memory(config, quant_type, context_len)
    
    # Gradient and optimizer memory (for training)
    grad_opt_memory = 0
    if training_type:
        grad_opt_memory = get_grad_opt_memory(
            training_type=training_type,
            optimizer=optimizer,
            quant_type=quant_type,
            model_size=model_size_params,
            config=config,
            context_len=context_len,
            batch_size=batch_size
        )
        grad_opt_memory = grad_opt_memory / (1024 * 1024)  # Convert to MB
    
    # CUDA overhead
    cuda_overhead = 650 #This can vary. This is just a rough estimate
    
    # Total memory
    total_memory = (
        model_size_mb +
        (activation_memory * batch_size) +
        extra_memory +
        grad_opt_memory +
        cuda_overhead
    )
    
    
    result =  {
        "Total": math.ceil(total_memory),
        "Model Size": math.ceil(model_size_mb),
        "Activation Memory": math.ceil(activation_memory * batch_size),
        "Grad & Optimizer memory": math.ceil(grad_opt_memory),
        "cuda + other overhead": cuda_overhead
    }
    return result
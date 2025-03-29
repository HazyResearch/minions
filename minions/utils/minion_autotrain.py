import os
import json
import glob
import argparse
import numpy as np
import torch
import platform
import psutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

# We make use of Hugging Face for model finetuning
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import datasets


class TrainingProtocol(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL_FINETUNE = "full_finetune"


@dataclass
class HardwareSpec:
    """Hardware specification for training."""
    device_type: str  # "cuda", "mps (Apple Silicon backend)", "cpu"
    device_name: str
    vram_gb: Optional[float] = None
    ram_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count(logical=False) or 1)
    
    @classmethod
    def detect(cls) -> "HardwareSpec":
        """Detect hardware specifications."""
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon"
            vram_gb = None  # MPS doesn't expose VRAM info
        else:
            device_type = "cpu"
            device_name = platform.processor() or "Unknown CPU"
            vram_gb = None
            
        return cls(
            device_type=device_type,
            device_name=device_name,
            vram_gb=vram_gb,
        )


@dataclass
class ModelCandidate:
    """Model candidate for training."""
    model_id: str
    context_length: int
    parameters_b: float  # billions of parameters
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_id: str
    protocol: TrainingProtocol
    batch_size: int
    learning_rate: float
    num_epochs: int
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    quantization: Optional[str] = None  # "4bit", "8bit", None
    bf16: bool = False
    gradient_accumulation_steps: int = 1


class MinionDatasetProcessor:
    """Process minion dataset for training."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        
    def load_logs(self) -> List[Dict[str, Any]]:
        """Load all minion logs from the log directory."""
        log_files = glob.glob(os.path.join(self.log_dir, "*.json"))
        logs = []
        
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                    logs.append(log_data)
            except Exception as e:
                print(f"Error loading {log_file}: {e}")
                
        return logs
    
    def extract_training_pairs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract training pairs from minion logs.
        
        We only extract worker/local model responses as assistant outputs.
        """
        training_pairs = []
        
        for log in logs:
            conversation = log.get("conversation", [])
            
            for i in range(len(conversation)):
                # Find entries where user is "local" and output is not None
                if conversation[i].get("user") == "local" and conversation[i].get("output"):
                    user_prompt = conversation[i].get("prompt", "")
                    assistant_response = conversation[i].get("output", "")
                    
                    if user_prompt and assistant_response:
                        training_pairs.append({
                            "prompt": user_prompt,
                            "response": assistant_response
                        })
                    
        return training_pairs
    
    def create_dataset(self, training_pairs: List[Dict[str, str]], tokenizer) -> datasets.Dataset:
        """Create a dataset from training pairs."""
        # Format for training: instruction, context (optional), response
        formatted_data = []
        
        for pair in training_pairs:
            # Format the text as per model requirements
            text = f"<s>[INST] {pair['prompt']} [/INST] {pair['response']}</s>"
            
            # Tokenize the text
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None,  # Return as list, not tensors
            )
            
            # Add labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            formatted_data.append(tokenized)
            
        return datasets.Dataset.from_list(formatted_data)


class ModelSelector:
    """Select the best model for training based on hardware specs."""
    
    # We hardcode a minimum memory requirement of double the model parameters and a recommended memory requirement of 4x the model parameters
    MODELS = [
        ModelCandidate(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            context_length=2048,
            parameters_b=1.1,
            requirements={"min_vram_gb": 2.5, "recommended_vram_gb": 5.0}
        ),
        ModelCandidate(
            model_id="meta-llama/Llama-3.2-1B",
            context_length=4096,
            parameters_b=1.0,
            requirements={"min_vram_gb": 2.0, "recommended_vram_gb": 4.0}
        ),
        ModelCandidate(
            model_id="meta-llama/Llama-3.2-3B",
            context_length=4096,
            parameters_b=3.0,
            requirements={"min_vram_gb": 6.0, "recommended_vram_gb": 12.0}
        ),
        ModelCandidate(
            model_id="microsoft/phi-4",
            context_length=8192,
            parameters_b=14.7,
            requirements={"min_vram_gb": 29.4, "recommended_vram_gb": 58.8}
        )
    ]
    
    @classmethod
    def select_model(cls, hardware: HardwareSpec) -> ModelCandidate:
        """Select the best model based on hardware specs."""
        if hardware.device_type == "cuda":
            # For CUDA, choose based on VRAM size
            available_models = [
                model for model in cls.MODELS 
                if hardware.vram_gb and hardware.vram_gb >= model.requirements.get("min_vram_gb", 0)
            ]
            
            if not available_models:
                # Fallback to smallest model
                available_models = [min(cls.MODELS, key=lambda m: m.parameters_b)]
                
            # Here we choose the largest model (as a proxy for most powerful model) that fits in device memory
            return max(available_models, key=lambda m: m.parameters_b)
            
        elif hardware.device_type == "mps":
            # We select the largest model conservatively given that only 75% of system RAM is available to MPS
            if hardware.ram_gb >= 24:
                candidates = [m for m in cls.MODELS if m.parameters_b <= 7.0]
                return max(candidates, key=lambda m: m.parameters_b)
            else:
                # For base Apple Silicon models
                candidates = [m for m in cls.MODELS if m.parameters_b <= 2.7]
                return max(candidates, key=lambda m: m.parameters_b)
                
        else:
            candidates = [m for m in cls.MODELS if m.parameters_b <= 1.1]
            return min(cls.MODELS, key=lambda m: m.parameters_b)


class TrainingProtocolSelector:
    """Select the best training protocol based on hardware specs and model."""
    
    @classmethod
    def select_protocol(cls, hardware: HardwareSpec, model: ModelCandidate) -> TrainingProtocol:
        """Select the best training protocol."""
        if hardware.device_type == "cuda":
            # After querying for hardware resources we attempt a full parameter finetune, then use LoRA, then QLoRA
            if hardware.vram_gb and hardware.vram_gb >= model.requirements.get("recommended_vram_gb", float('inf')):
                # If we have recommended VRAM, we can do full finetuning
                return TrainingProtocol.FULL_FINETUNE
            elif hardware.vram_gb and hardware.vram_gb >= model.requirements.get("min_vram_gb", float('inf')) * 0.6:
                # If we have enough VRAM for 4-bit quantization
                return TrainingProtocol.QLORA
                
        elif hardware.device_type == "mps":
            # MPS works best with QLoRA for larger models
            if model.parameters_b > 2.0:
                return TrainingProtocol.QLORA
            else:
                return TrainingProtocol.LORA
                
        else:
            # For CPU, always use QLoRA 4-bit quantization
            return TrainingProtocol.QLORA


class HyperparameterOptimizer:
    """Optimize hyperparameters for training."""
    
    @classmethod
    def get_optimal_hyperparams(
        cls, 
        hardware: HardwareSpec, 
        model: ModelCandidate, 
        protocol: TrainingProtocol,
        dataset_size: int
    ) -> TrainingConfig:
        """Get optimal hyperparameters for training."""
        # Base configuration
        config = {
            "model_id": model.model_id,
            "protocol": protocol,
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
        }
        
        # We use common LoRA / QLoRA hyperparameters
        if hardware.device_type == "cuda":
            if protocol == TrainingProtocol.FULL_FINETUNE:
                config["batch_size"] = cls._get_batch_size(hardware, model, full_finetune=True)
                config["bf16"] = True
            elif protocol == TrainingProtocol.QLORA:
                config["batch_size"] = cls._get_batch_size(hardware, model)
                config["lora_r"] = 64
                config["lora_alpha"] = 128
                config["lora_dropout"] = 1
                config["quantization"] = "4bit"
                config["bf16"] = True
            else:  # LoRA
                config["batch_size"] = cls._get_batch_size(hardware, model)
                config["lora_r"] = 32
                config["lora_alpha"] = 64
                config["lora_dropout"] = 0.1
                config["bf16"] = True
                
        elif hardware.device_type == "mps":
            # MPS (Apple Silicon)
            if protocol == TrainingProtocol.QLORA:
                config["batch_size"] = 2 if model.parameters_b > 2.0 else 4
                config["lora_r"] = 32
                config["lora_alpha"] = 64
                config["lora_dropout"] = 0.05
                config["quantization"] = "4bit"
                # MPS doesn't support bf16 training
                config["bf16"] = False
            else:
                config["batch_size"] = 4 if model.parameters_b > 2.0 else 8
                config["lora_r"] = 16
                config["lora_alpha"] = 32
                config["lora_dropout"] = 0.1
                config["bf16"] = False
                
        else:
            # CPU - very conservative settings
            config["batch_size"] = 1
            config["lora_r"] = 8
            config["lora_alpha"] = 16
            config["lora_dropout"] = 0.1
            config["gradient_accumulation_steps"] = 8
            config["bf16"] = False
            
        # We stick to 3 epochs for finetuning by default following common empirical results
        config["num_epochs"] = 3
            
        # If gradient_accumulation_steps is too small, increase it
        if config["batch_size"] == 1:
            config["gradient_accumulation_steps"] = max(4, config["gradient_accumulation_steps"])
            
        return TrainingConfig(**config)
    
    @staticmethod
    def _get_batch_size(hardware: HardwareSpec, model: ModelCandidate, full_finetune: bool = False) -> int:
        """Get optimal batch size based on available VRAM."""
        if not hardware.vram_gb:
            return 1
            
        vram_gb = hardware.vram_gb
        param_b = model.parameters_b
        
        if full_finetune:
            # Full finetuning is very VRAM intensive
            if vram_gb > 40:  # A100 class GPU or better
                return 8
            elif vram_gb > 24:
                return 4
            elif vram_gb > 16:
                return 2
            else:
                return 1
        else:
            # LoRA/QLoRA is more VRAM efficient
            if param_b > 6:  # 7B+ models
                if vram_gb > 24:
                    return 8
                elif vram_gb > 16:
                    return 4
                elif vram_gb > 10:
                    return 2
                else:
                    return 1
            else:
                if vram_gb > 16:
                    return 16
                elif vram_gb > 8:
                    return 8
                elif vram_gb > 4:
                    return 4
                else:
                    return 2


class MinionTrainer:
    """Train a minion model with the optimal configuration."""
    
    def __init__(
        self, 
        config: TrainingConfig,
        dataset_processor: MinionDatasetProcessor,
        output_dir: str = "output",
        hub_token: Optional[str] = None,
    ):
        self.config = config
        self.dataset_processor = dataset_processor
        self.output_dir = output_dir
        self.hub_token = hub_token
        
    def prepare_model_and_tokenizer(self):
        """Prepare model and tokenizer for training."""
        print(f"Loading model: {self.config.model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Prepare model with appropriate quantization if needed
        if self.config.quantization == "4bit":
            compute_dtype = torch.float16 if not self.config.bf16 else torch.bfloat16
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            
        elif self.config.quantization == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                load_in_8bit=True,
                trust_remote_code=True,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.config.bf16 else torch.bfloat16,
            )
            
        # Apply LoRA if needed
        if self.config.protocol in [TrainingProtocol.LORA, TrainingProtocol.QLORA]:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self._get_target_modules(model),
                bias="none",
            )
            
            model = get_peft_model(model, peft_config)
            
        return model, tokenizer
    
    def train(self):
        """Train the model."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and process dataset
        logs = self.dataset_processor.load_logs()
        print(f"Loaded {len(logs)} log files")
        
        training_pairs = self.dataset_processor.extract_training_pairs(logs)
        print(f"Extracted {len(training_pairs)} training examples")
        
        if not training_pairs:
            print("No training examples found. Exiting.")
            return
            
        # Prepare model and tokenizer
        model, tokenizer = self.prepare_model_and_tokenizer()
        
        # Create dataset
        dataset = self.dataset_processor.create_dataset(training_pairs, tokenizer)
        
        # Split dataset with handling for very small datasets
        if len(dataset) <= 2:  # For extremely small datasets
            # Use the same data for training and testing
            dataset_dict = {
                "train": dataset,
                "test": dataset
            }
        else:
            # For larger datasets, use proper split
            test_size = min(0.1, max(1 / len(dataset), 1 / 10))  # Adaptive test size
            dataset_dict = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
        
        print(f"Training set size: {len(dataset_dict['train'])}")
        print(f"Test set size: {len(dataset_dict['test'])}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            warmup_steps=100,
            push_to_hub=bool(self.hub_token),
            hub_token=self.hub_token,
            # Handle mixed precision settings based on device type
            bf16=self.config.bf16 and torch.cuda.is_available(),  # Only use bf16 on CUDA
            fp16=not self.config.bf16 and torch.cuda.is_available(),  # Only use fp16 on CUDA
            load_best_model_at_end=True,
            report_to="tensorboard",
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            data_collator=data_collator,
        )
        
        # Train model
        print(f"Starting training with {self.config}")
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training complete. Model saved to {self.output_dir}")
        
    def _get_target_modules(self, model):
        """Get target modules for LoRA based on model architecture."""
        # This is a heuristic approach; may need to be adjusted for specific models
        model_name = self.config.model_id.lower()
        
        # Q, K, V, O are common target modules for LoRA, it may sometimes be worthwhile to target other modules such as lm_head
        if "mistral" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "llama" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "phi" in model_name:
            return ["Wqkv", "out_proj"]
        else:
            # Default target modules
            return ["query", "key", "value", "attention.output.dense"]


def auto_train_minion(
    log_dir: str = "minion_logs",
    output_dir: str = "trained_minion",
    hub_model_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    model_name: Optional[str] = None
):
    """Automatically train a minion model with optimal settings."""
    print("Starting minion auto-training...")
    
    # Detect hardware
    hardware = HardwareSpec.detect()
    print(f"Detected hardware: {hardware.device_type} device {hardware.device_name} found")
    if hardware.vram_gb:
        print(f"VRAM: {hardware.vram_gb:.2f} GB")
    print(f"RAM: {hardware.ram_gb:.2f} GB")
    print(f"CPU cores: {hardware.cpu_count}")
    
    # Initialize dataset processor
    dataset_processor = MinionDatasetProcessor(log_dir=log_dir)
    
    # Load and analyze dataset
    logs = dataset_processor.load_logs()
    training_pairs = dataset_processor.extract_training_pairs(logs)
    dataset_size = len(training_pairs)
    print(f"Dataset size: {dataset_size} examples")
    
    if dataset_size == 0:
        print("No training examples found. Exiting.")
        return
    
    # Select optimal model
    if model_name == "default":
        model = ModelSelector.MODELS[0] # default to TinyLlama-1.1B-Chat-v1.0 for resource efficiency and no huggingface token required
    else:
        model = ModelSelector.select_model(hardware, model_name)
    print(f"Selected model: {model.model_id} ({model.parameters_b}B parameters)")
    
    # Select optimal training protocol
    protocol = TrainingProtocolSelector.select_protocol(hardware, model)
    print(f"Selected training protocol: {protocol.value}")
    
    # Get optimal hyperparameters
    config = HyperparameterOptimizer.get_optimal_hyperparams(
        hardware, model, protocol, dataset_size
    )
    print("Training configuration:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    
    if protocol in [TrainingProtocol.LORA, TrainingProtocol.QLORA]:
        print(f"  - LoRA rank: {config.lora_r}")
        print(f"  - LoRA alpha: {config.lora_alpha}")
        print(f"  - LoRA dropout: {config.lora_dropout}")
        
    if protocol == TrainingProtocol.QLORA:
        print(f"  - Quantization: {config.quantization}")
    
    # Initialize trainer
    trainer = MinionTrainer(
        config=config,
        dataset_processor=dataset_processor,
        output_dir=output_dir,
        hub_token=hub_token if hub_model_id else None,
    )
    
    # Train model
    trainer.train()
    
    print("Auto-training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-train a minion model")
    parser.add_argument("--log-dir", type=str, default="minion_logs", help="Directory containing minion logs")
    parser.add_argument("--output-dir", type=str, default="trained_minion", help="Output directory for trained model")
    parser.add_argument("--hub-model-id", type=str, help="Hugging Face Hub model ID for pushing")
    parser.add_argument("--hub-token", type=str, help="Hugging Face Hub token for pushing")
    parser.add_argument("--model-name", type=str, default="auto", help="Pass 'default' to use TinyLlama-1.1B-Chat-v1.0, or 'auto' to select the best model for your hardware")
    
    args = parser.parse_args()
    
    auto_train_minion(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        model_name=args.model_name
    ) 
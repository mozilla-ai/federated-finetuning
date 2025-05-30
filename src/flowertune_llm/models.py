import math
import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with efficient quantization and LoRA tuning.

    Uses smaller models like Mistral-7B, Phi-2, or TinyLLaMA to optimize efficiency.
    """

    # Suggested small model options
    model_choices = {"qwen2-0.5": "Qwen/Qwen2-0.5B"}

    # Choose the model (default: Mistral-7B)
    model_name = model_choices.get(model_cfg.name.lower(), model_cfg.name)

    # Handle different quantization settings
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype = torch.bfloat16
        device_map = "auto"
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.bfloat16
        device_map = "auto"
    elif model_cfg.quantization == 0:
        quantization_config = None  # No quantization
        torch_dtype = torch.float32  # Ensure compatibility with CPU training
        device_map = "cpu"  # Force model to run on CPU
    else:
        raise ValueError(
            f"Use 4-bit, 8-bit, or disable quantization (0). You passed: {model_cfg.quantization}"
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # LoRA Configuration (Optimized for Small Models)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Add target LoRA layers
    )

    return get_peft_model(model, peft_config)


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

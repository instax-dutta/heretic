# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
FSDP (Fully Sharded Data Parallel) utilities for TPU model parallelism.

This module provides helpers for wrapping models with FSDP on PyTorch/XLA
for multi-core TPU training and inference.
"""

import json
import os
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .system import detect_tpu, _is_torch_xla_available


# Architecture-specific transformer layer class names for FSDP wrapping
# These are the module classes that should be wrapped as FSDP units
TRANSFORMER_LAYER_CLASSES = {
    # Llama family
    "LlamaForCausalLM": ["LlamaDecoderLayer"],
    "LlamaForSequenceClassification": ["LlamaDecoderLayer"],
    "CodeLlamaForCausalLM": ["LlamaDecoderLayer"],
    # Qwen family
    "Qwen2ForCausalLM": ["Qwen2DecoderLayer"],
    "Qwen2MoeForCausalLM": ["Qwen2MoeDecoderLayer"],
    "Qwen3ForCausalLM": ["Qwen3DecoderLayer"],
    "Qwen3MoeForCausalLM": ["Qwen3MoeDecoderLayer"],
    # Gemma family
    "GemmaForCausalLM": ["GemmaDecoderLayer"],
    "Gemma2ForCausalLM": ["Gemma2DecoderLayer"],
    "Gemma3ForCausalLM": ["Gemma3DecoderLayer"],
    "Gemma3ForConditionalGeneration": ["Gemma3DecoderLayer"],
    # Mistral family
    "MistralForCausalLM": ["MistralDecoderLayer"],
    "MixtralForCausalLM": ["MixtralDecoderLayer"],
    # Phi family
    "Phi3ForCausalLM": ["Phi3DecoderLayer"],
    "Phi4ForCausalLM": ["Phi4DecoderLayer"],
    # Yi
    "YiForCausalLM": ["YiDecoderLayer"],
    # InternLM
    "InternLM2ForCausalLM": ["InternLM2DecoderLayer"],
    "InternLM2ForSequenceClassification": ["InternLM2DecoderLayer"],
    # Granite
    "GraniteForCausalLM": ["GraniteDecoderLayer"],
    "GraniteMoeForCausalLM": ["GraniteMoeDecoderLayer"],
    # Command-R
    "CohereForCausalLM": ["CohereDecoderLayer"],
    # SmolLM
    "SmolLM3ForCausalLM": ["SmolLM3DecoderLayer"],
    # DeepSeek
    "DeepseekV2ForCausalLM": ["DeepseekV2DecoderLayer"],
    "DeepseekV3ForCausalLM": ["DeepseekV3DecoderLayer"],
}


def get_fsdp_layer_class(model: PreTrainedModel) -> list[str]:
    """Get the transformer layer class names for FSDP wrapping based on model architecture."""
    model_class = model.__class__.__name__
    return TRANSFORMER_LAYER_CLASSES.get(model_class, [])


def get_default_fsdp_config(model: PreTrainedModel) -> dict[str, Any]:
    """Generate a default FSDP configuration for the given model."""
    layer_classes = get_fsdp_layer_class(model)
    
    config = {
        "xla": True,
        "xla_fsdp_settings": {
            "min_num_params": 100_000_000,  # 100M params minimum for sharding
            "grad_ckpt": True,
        },
        "fsdp_transformer_layer_cls_to_wrap": layer_classes,
    }
    
    return config


def save_fsdp_config(config: dict[str, Any], path: str) -> None:
    """Save FSDP configuration to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_fsdp_config(path: str) -> dict[str, Any]:
    """Load FSDP configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def wrap_model_fsdp(
    model: PreTrainedModel,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PreTrainedModel:
    """
    Wrap a model with FSDP for TPU model parallelism.
    
    Args:
        model: The model to wrap
        config: FSDP configuration dict (or path to JSON file)
        **kwargs: Additional arguments passed to FSDP wrapper
        
    Returns:
        FSDP-wrapped model
    """
    if not detect_tpu():
        raise RuntimeError("FSDP wrapping only supported on TPU")
    
    if not _is_torch_xla_available():
        raise RuntimeError("torch_xla not available")
    
    import torch_xla.distributed.fsdp as xla_fsdp
    import torch_xla.core.xla_model as xm
    
    # Load config if path provided
    if isinstance(config, str):
        config = load_fsdp_config(config)
    elif config is None:
        config = get_default_fsdp_config(model)
    
    # Get layer classes to wrap
    layer_classes = config.get("fsdp_transformer_layer_cls_to_wrap", [])
    if not layer_classes:
        layer_classes = get_fsdp_layer_class(model)
    
    if not layer_classes:
        raise ValueError(f"No transformer layer classes found for model {model.__class__.__name__}")
    
    # Get XLA FSDP settings
    xla_fsdp_settings = config.get("xla_fsdp_settings", {})
    
    # Wrap the model
    fsdp_kwargs = {
        "min_num_params": xla_fsdp_settings.get("min_num_params", 100_000_000),
        "auto_wrap_policy": xla_fsdp.transformer_auto_wrap_policy,
        "transformer_layer_cls": layer_classes,
    }
    
    if xla_fsdp_settings.get("grad_ckpt", False):
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
    
    # Apply FSDP wrapping
    model = xla_fsdp.wrap(model, **fsdp_kwargs)
    
    return model


def unwrap_fsdp_model(model: PreTrainedModel) -> PreTrainedModel:
    """Unwrap a FSDP-wrapped model."""
    if not _is_torch_xla_available():
        return model
    
    import torch_xla.distributed.fsdp as xla_fsdp
    
    if isinstance(model, xla_fsdp.FSDP):
        return model._fsdp_wrapped_module
    
    return model


def is_fsdp_model(model: PreTrainedModel) -> bool:
    """Check if model is wrapped with FSDP."""
    if not _is_torch_xla_available():
        return False
    
    import torch_xla.distributed.fsdp as xla_fsdp
    
    return isinstance(model, xla_fsdp.FSDP)


# Default FSDP config for TPU v5e-8
DEFAULT_FSDP_CONFIG_V5E_8 = {
    "xla": True,
    "xla_fsdp_settings": {
        "min_num_params": 100_000_000,
        "grad_ckpt": True,
    },
    "fsdp_transformer_layer_cls_to_wrap": [],  # Populated dynamically
}
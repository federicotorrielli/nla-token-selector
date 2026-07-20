"""Resolve the residual-stream layer module to hook, across model families."""

from __future__ import annotations

import torch


def _resolve_layer_module(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    """Walk the Llama-family path `model.model.layers[K]`. Same assumption as the
    NLA extractor (Qwen/Llama/Mistral/Gemma all share this path). Gemma 3 wraps the
    decoder under `language_model`."""
    base = model.model if hasattr(model, "model") else model
    if hasattr(base, "language_model"):
        base = base.language_model
    if hasattr(base, "model"):
        base = base.model
    layers = base.layers
    assert 0 <= layer_index < len(layers), (
        f"layer_index={layer_index} out of range for model with {len(layers)} layers"
    )
    return layers[layer_index]

"""
Parameter Classification for Qwen2.5-VL Model

This module classifies Qwen2.5-VL parameters into three categories for evolutionary strategies:
- 'lora': Apply low-rank perturbations (memory efficient, ~100× savings)
- 'full': Apply full-rank Gaussian noise (for small parameters)
- 'frozen': No evolution (vision encoder, embeddings)

The classification balances:
1. Memory efficiency (low-rank for large matrices)
2. Training stability (freeze pretrained vision encoder)
3. Expressiveness (full-rank for critical small parameters)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from config import FREEZE_VISION, SMALL_PARAM_THRESHOLD


# Parameter classification constants
FROZEN = 0  # No evolution
FULL = 1    # Full-rank Gaussian noise
LORA = 2    # Low-rank LoRA-style perturbation


def get_qwen_es_classification(model: nn.Module) -> Dict[str, int]:
    """
    Classify Qwen2.5-VL parameters for evolutionary strategies.

    Args:
        model: Qwen2.5-VL model instance

    Returns:
        Dictionary mapping parameter name to classification:
        - 0 (FROZEN): No evolution
        - 1 (FULL): Full-rank noise
        - 2 (LORA): Low-rank noise

    Classification Rules:
    --------------------
    FROZEN (no evolution):
        - Vision encoder (if FREEZE_VISION=True)
        - Embedding layers
        - Output projection head
        - Position embeddings

    LORA (low-rank perturbations):
        - Attention projections (Q, K, V, O)
        - MLP/FFN layers (gate, up, down)
        - Any large matrices (>SMALL_PARAM_THRESHOLD elements)

    FULL (full-rank noise):
        - LayerNorm parameters
        - Biases
        - Small parameters (<SMALL_PARAM_THRESHOLD elements)

    Example Parameter Names:
    -----------------------
    Frozen:
        - visual.* (vision encoder)
        - model.embed_tokens.weight
        - lm_head.weight

    LoRA:
        - model.layers[*].self_attn.q_proj.weight
        - model.layers[*].self_attn.k_proj.weight
        - model.layers[*].self_attn.v_proj.weight
        - model.layers[*].self_attn.o_proj.weight
        - model.layers[*].mlp.gate_proj.weight
        - model.layers[*].mlp.up_proj.weight
        - model.layers[*].mlp.down_proj.weight

    Full:
        - model.layers[*].input_layernorm.weight
        - model.layers[*].post_attention_layernorm.weight
        - *.bias (all biases)
    """
    classification = {}

    for name, param in model.named_parameters():
        # Skip non-leaf parameters (shouldn't happen, but be safe)
        if not param.requires_grad:
            classification[name] = FROZEN
            continue

        # Rule 1: Vision encoder - always freeze if FREEZE_VISION=True
        if FREEZE_VISION and 'visual' in name.lower():
            classification[name] = FROZEN

        # Rule 2: Embeddings - freeze (pretrained, stable)
        elif 'embed' in name.lower():
            classification[name] = FROZEN

        # Rule 3: Output head - freeze (pretrained vocabulary)
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            classification[name] = FROZEN

        # Rule 4: Position embeddings - freeze
        elif 'position' in name.lower() or 'pos_emb' in name.lower():
            classification[name] = FROZEN

        # Rule 5: Attention projections - low-rank (large, critical)
        elif any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            classification[name] = LORA

        # Rule 6: MLP/FFN layers - low-rank (large, critical)
        elif any(mlp in name for mlp in ['gate_proj', 'up_proj', 'down_proj']):
            classification[name] = LORA

        # Rule 7: LayerNorm - full-rank (small, important for stability)
        elif 'norm' in name.lower() or 'ln' in name.lower():
            classification[name] = FULL

        # Rule 8: Biases - full-rank (small)
        elif 'bias' in name.lower():
            classification[name] = FULL

        # Rule 9: Large parameters - low-rank (memory efficiency)
        elif param.numel() >= SMALL_PARAM_THRESHOLD:
            classification[name] = LORA

        # Rule 10: Small parameters - full-rank (expressiveness)
        elif param.numel() < SMALL_PARAM_THRESHOLD:
            classification[name] = FULL

        # Default: freeze (conservative)
        else:
            classification[name] = FROZEN

    return classification


def get_evolvable_params_info(model: nn.Module,
                               classification: Dict[str, int]) -> Dict[str, Dict]:
    """
    Extract information about evolvable parameters for noise generation.

    Args:
        model: Qwen2.5-VL model
        classification: Parameter classification dict

    Returns:
        Dictionary mapping parameter name to info dict:
        {
            'shape': parameter shape tuple,
            'classification': FROZEN/FULL/LORA,
            'numel': number of elements,
            'dtype': parameter dtype,
            'device': parameter device
        }
    """
    evolvable_params = {}

    for name, param in model.named_parameters():
        cls = classification.get(name, FROZEN)

        # Only include evolvable parameters
        if cls in [FULL, LORA]:
            evolvable_params[name] = {
                'shape': tuple(param.shape),
                'classification': cls,
                'numel': param.numel(),
                'dtype': param.dtype,
                'device': param.device
            }

    return evolvable_params


def print_classification_summary(model: nn.Module,
                                  classification: Dict[str, int],
                                  verbose: bool = True):
    """
    Print summary statistics of parameter classification.

    Args:
        model: Qwen2.5-VL model
        classification: Parameter classification dict
        verbose: If True, print detailed breakdown
    """
    # Count parameters by classification
    frozen_params = 0
    full_params = 0
    lora_params = 0
    frozen_count = 0
    full_count = 0
    lora_count = 0

    for name, param in model.named_parameters():
        cls = classification.get(name, FROZEN)
        numel = param.numel()

        if cls == FROZEN:
            frozen_params += numel
            frozen_count += 1
        elif cls == FULL:
            full_params += numel
            full_count += 1
        elif cls == LORA:
            lora_params += numel
            lora_count += 1

    total_params = frozen_params + full_params + lora_params

    print("=" * 80)
    print("Parameter Classification Summary")
    print("=" * 80)
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print()
    print(f"Frozen (no evolution):")
    print(f"  Count: {frozen_count:,} tensors")
    print(f"  Parameters: {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)")
    print()
    print(f"Full-rank noise:")
    print(f"  Count: {full_count:,} tensors")
    print(f"  Parameters: {full_params:,} ({full_params / total_params * 100:.1f}%)")
    print()
    print(f"Low-rank noise (LoRA):")
    print(f"  Count: {lora_count:,} tensors")
    print(f"  Parameters: {lora_params:,} ({lora_params / total_params * 100:.1f}%)")
    print()

    evolvable_params = full_params + lora_params
    print(f"Total evolvable: {evolvable_params:,} ({evolvable_params / total_params * 100:.1f}%)")
    print("=" * 80)

    if verbose:
        print("\nDetailed Breakdown:")
        print("-" * 80)

        # Group by classification
        for cls_type, cls_name in [(FROZEN, "FROZEN"), (FULL, "FULL"), (LORA, "LORA")]:
            params_in_cls = [
                (name, param.numel())
                for name, param in model.named_parameters()
                if classification.get(name, FROZEN) == cls_type
            ]

            if params_in_cls:
                print(f"\n{cls_name} ({len(params_in_cls)} tensors):")
                # Show first 10 and last 3
                for name, numel in params_in_cls[:10]:
                    print(f"  {name}: {numel:,}")
                if len(params_in_cls) > 13:
                    print(f"  ... ({len(params_in_cls) - 13} more) ...")
                for name, numel in params_in_cls[-3:]:
                    print(f"  {name}: {numel:,}")

        print("-" * 80)


def estimate_memory_savings(model: nn.Module,
                             classification: Dict[str, int],
                             rank: int = 2) -> Tuple[float, float]:
    """
    Estimate memory savings from low-rank perturbations.

    Args:
        model: Qwen2.5-VL model
        classification: Parameter classification dict
        rank: Low-rank dimension

    Returns:
        Tuple of (full_rank_memory_gb, low_rank_memory_gb)

    Memory Calculation:
    ------------------
    Full-rank noise for matrix (a, b): a × b × 4 bytes (float32)
    Low-rank noise for matrix (a, b): (a × r + b × r) × 4 bytes
    Savings ratio: (a × b) / (a × r + b × r) ≈ min(a,b) / (2r)
    """
    full_rank_memory = 0
    low_rank_memory = 0

    for name, param in model.named_parameters():
        cls = classification.get(name, FROZEN)

        if cls == LORA and len(param.shape) == 2:
            # Matrix parameter
            a, b = param.shape
            full_rank_memory += a * b * 4  # 4 bytes per float32
            low_rank_memory += (a * rank + b * rank) * 4

        elif cls == FULL:
            # Full-rank noise
            numel = param.numel()
            full_rank_memory += numel * 4
            low_rank_memory += numel * 4

    # Convert to GB
    full_rank_gb = full_rank_memory / (1024 ** 3)
    low_rank_gb = low_rank_memory / (1024 ** 3)

    return full_rank_gb, low_rank_gb


if __name__ == "__main__":
    """
    Test parameter classification on Qwen2.5-VL model.

    Note: This requires the model to be downloaded at config.MODEL_PATH.
    If the model is not available, this will fail.
    """
    import sys
    from config import MODEL_PATH

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        print(f"Loading model from {MODEL_PATH}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Load to CPU for classification
        )

        print("Classifying parameters...")
        classification = get_qwen_es_classification(model)

        print_classification_summary(model, classification, verbose=True)

        print("\nMemory Savings Estimation:")
        for rank in [1, 2, 4, 8]:
            full_mem, low_mem = estimate_memory_savings(model, classification, rank)
            savings = (1 - low_mem / full_mem) * 100
            print(f"  Rank {rank}: {full_mem:.2f} GB → {low_mem:.2f} GB ({savings:.1f}% savings)")

        print("\n✓ Parameter classification completed successfully")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nThis test requires Qwen2.5-VL model to be available.")
        print(f"Expected location: {MODEL_PATH}")
        sys.exit(1)

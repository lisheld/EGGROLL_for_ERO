"""
Configuration file for ERO_EGGROLL: EGGROLL-based training for ARC-1 reasoning tasks.

This module contains all hyperparameters, paths, and settings for training
Qwen2.5-VL-7B on ARC visual reasoning tasks using evolutionary strategies.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = str(BASE_DIR / "ERO" / "models" / "base_model")
MODEL_NAME = "Qwen2.5-VL-7B"

# Data paths
DATA_PATH = str(BASE_DIR / "ERO" / "data")

# ARC-1 task IDs (15 visual reasoning tasks)
ARC_TASKS = [
    "351d6448", "414297c0", "e6de6e8f", "e7a25a18", "505fff84",
    "b1fc8b8e", "1a6449f1", "3194b014", "9b4c17c4", "0a1d4ef5",
    "9a4bb226", "12422b43", "1c02dbbe", "477d2879", "67b4a34d"
]

# Output paths
CHECKPOINT_DIR = str(BASE_DIR / "ERO_EGGROLL" / "checkpoints")
RESULTS_DIR = str(BASE_DIR / "ERO_EGGROLL" / "results")

# =============================================================================
# Evolution Hyperparameters
# =============================================================================

# Number of training epochs
NUM_EPOCHS = 12

# Population size (number of model variants sampled per epoch)
# Start with 32 for memory safety, can increase to 64 or 128
POPULATION_SIZE = 32  # Options: 32, 64, 128

# Noise scale (controls magnitude of parameter perturbations)
# Smaller values = smaller steps, more stable but slower learning
# Larger values = larger steps, faster but potentially unstable
SIGMA = 0.01  # Range: 0.001 - 0.1

# Low-rank dimension for EGGROLL noise
# rank=1: maximum compression, least expressive
# rank=2: good balance (default)
# rank=4,8: more expressive, less compression
RANK = 2  # Options: 1, 2, 4, 8

# Learning rate scale multiplier
# Final lr = LR_SCALE * sigma^2 * sqrt(population_size) (standard ES formula)
LR_SCALE = 1.0  # Range: 0.1 - 10.0

# Noise reuse factor
# 0 or 1: No noise reuse (generate new noise each epoch)
# >1: Reuse same noise pattern every NOISE_REUSE epochs
NOISE_REUSE = 1  # Options: 0, 1, 2, 5, 10

# =============================================================================
# Memory Optimization
# =============================================================================

# Freeze vision encoder (recommended for memory + stability)
FREEZE_VISION = True

# Enable gradient checkpointing (trades compute for memory)
GRADIENT_CHECKPOINTING = True

# Use mixed precision (bfloat16)
MIXED_PRECISION = True

# Model offloading (move base model to CPU, only load perturbed variant to GPU)
# Enable this if you run out of GPU memory
MODEL_OFFLOADING = False

# Freeze non-LoRA parameters (only evolve low-rank adapted layers)
# Recommended: True (saves memory and compute)
FREEZE_NONLORA = True

# =============================================================================
# Model Evaluation
# =============================================================================

# Maximum number of new tokens to generate
# ARC answers are typically 100-500 characters
MAX_NEW_TOKENS = 500  # Range: 300-1000

# Sampling temperature (0.0 = greedy/deterministic, >0 = stochastic)
TEMPERATURE = 0.0

# Batch size for evaluation (how many ARC tasks to evaluate in parallel)
# Set to 1 for memory safety, can increase if memory allows
EVAL_BATCH_SIZE = 1

# =============================================================================
# Checkpointing & Logging
# =============================================================================

# Save checkpoint every N epochs
SAVE_EVERY = 3  # Options: 1, 3, 5

# Save best model whenever fitness improves
SAVE_BEST = True

# Verbose logging (print detailed information during training)
VERBOSE = True

# Log fitness scores to JSON every epoch
LOG_FITNESS = True

# =============================================================================
# Device Configuration
# =============================================================================

# Primary CUDA device
DEVICE = "cuda:0"

# JAX device (should match CUDA device)
JAX_DEVICE = "gpu:0"

# Random seeds
SEED = 42

# =============================================================================
# Advanced Settings (typically don't need to change)
# =============================================================================

# Degeneration threshold (from ERO)
# If fitness doesn't improve for this many epochs, reset to base model
DEGEN_THRESHOLD = 3

# Early stopping threshold
# Stop training if mean fitness across all tasks reaches this value
EARLY_STOP_FITNESS = 1.0  # 1.0 = perfect scores on all tasks

# Number of ARC training examples shown in prompt
NUM_TRAIN_EXAMPLES = 2  # ARC-1 format uses 2 examples

# Parameter classification thresholds
SMALL_PARAM_THRESHOLD = 10000  # Params with <10K elements get full-rank noise

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check paths exist
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model path does not exist: {MODEL_PATH}")
    if not os.path.exists(DATA_PATH):
        errors.append(f"Data path does not exist: {DATA_PATH}")

    # Create output directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Validate hyperparameters
    if POPULATION_SIZE < 1:
        errors.append(f"POPULATION_SIZE must be >= 1, got {POPULATION_SIZE}")
    if RANK < 1:
        errors.append(f"RANK must be >= 1, got {RANK}")
    if SIGMA <= 0:
        errors.append(f"SIGMA must be > 0, got {SIGMA}")
    if NUM_EPOCHS < 1:
        errors.append(f"NUM_EPOCHS must be >= 1, got {NUM_EPOCHS}")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    return True


# Run validation on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: {e}")


if __name__ == "__main__":
    """Print current configuration when run as script."""
    print("=" * 80)
    print("ERO_EGGROLL Configuration")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Data Path: {DATA_PATH}")
    print(f"\nEvolution Parameters:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Population Size: {POPULATION_SIZE}")
    print(f"  Sigma (noise scale): {SIGMA}")
    print(f"  Rank (low-rank dim): {RANK}")
    print(f"  Learning Rate Scale: {LR_SCALE}")
    print(f"\nMemory Optimization:")
    print(f"  Freeze Vision Encoder: {FREEZE_VISION}")
    print(f"  Gradient Checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"  Mixed Precision: {MIXED_PRECISION}")
    print(f"  Model Offloading: {MODEL_OFFLOADING}")
    print(f"\nOutput:")
    print(f"  Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"  Results Dir: {RESULTS_DIR}")
    print(f"  Save Every: {SAVE_EVERY} epochs")
    print("=" * 80)

    # Validate
    try:
        validate_config()
        print("\n✓ Configuration is valid")
    except ValueError as e:
        print(f"\n✗ Configuration validation failed:\n{e}")

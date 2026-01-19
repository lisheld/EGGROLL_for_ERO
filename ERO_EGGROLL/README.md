# ERO_EGGROLL: EGGROLL Framework for ARC-1 Visual Reasoning

This project integrates the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) framework with **Qwen2.5-VL-7B** to train a visual reasoning model on ARC-1 tasks using evolutionary strategies.

## Overview

**Goal:** Replace ERO's memory-intensive full-rank evolutionary strategies with EGGROLL's low-rank perturbations for single-GPU training.

**Key Innovation:**
- ERO: 1000 samples across 4 GPUs with full-rank noise (~256GB memory)
- EGGROLL: 32-128 samples on 1 GPU with low-rank noise (~24-48GB memory)
- **Memory savings: ~100× for large matrices**

## Architecture

### Hybrid PyTorch-JAX Design

- **PyTorch**: Qwen2.5-VL-7B model (complex vision-language architecture)
- **JAX**: EGGROLL noise generation (low-rank, memory efficient)
- **Bridge**: NumPy arrays for data transfer

### Parameter Classification

Parameters are classified into three categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **FROZEN** | No evolution | Vision encoder, embeddings, output head |
| **LORA** | Low-rank noise | Attention (Q,K,V,O), MLP (gate, up, down) |
| **FULL** | Full-rank noise | LayerNorm, biases, small parameters |

**Memory Impact** (hidden_dim=3584, rank=2):
- Low-rank noise: ~14KB per matrix
- Full-rank noise: ~49MB per matrix
- **Savings: ~3,500×**

## Installation

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU with 24GB+ VRAM (48GB+ recommended)
```

### Dependencies

From ERO:
```bash
cd ERO
pip install -r requirements.txt
```

From HyperscaleES:
```bash
cd HyperscaleES
pip install -e .
```

Additional:
```bash
pip install numpy transformers torch jax jaxlib
```

### Model Setup

Download Qwen2.5-VL-7B model to `ERO/models/base_model/`:

```bash
# Using HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ERO/models/base_model
```

## Project Structure

```
ERO_EGGROLL/
├── README.md                      # This file
├── config.py                      # Hyperparameters and settings
├── qwen_param_mapper.py           # Parameter classification system
├── arc_task.py                    # ARC-1 task adapter
├── arc_evaluator.py               # Fitness evaluation (TODO)
├── pytorch_jax_bridge.py          # JAX-PyTorch conversion (TODO)
├── arc_evolution_single_gpu.py    # Main training script (TODO)
├── utils.py                       # Helper functions (TODO)
├── checkpoints/                   # Saved model checkpoints
├── results/                       # Training logs and scores
└── tests/
    ├── test_components.py         # Unit tests (TODO)
    └── test_integration.py        # Integration tests (TODO)
```

## Configuration

Edit `config.py` to adjust hyperparameters:

```python
# Evolution parameters
NUM_EPOCHS = 12
POPULATION_SIZE = 32  # Start small, increase to 64-128
SIGMA = 0.01          # Noise scale
RANK = 2              # Low-rank dimension
LR_SCALE = 1.0        # Learning rate multiplier

# Memory optimization
FREEZE_VISION = True
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = True
```

View current configuration:
```bash
python config.py
```

## Usage

### 1. Test Components (Week 1)

```bash
# Test parameter classification
python qwen_param_mapper.py

# Test ARC task loading
python arc_task.py

# Test configuration
python config.py
```

### 2. Train Model (Week 3+)

```bash
# Coming soon: main training script
python arc_evolution_single_gpu.py
```

### 3. Evaluate Checkpoint

```bash
# Coming soon: evaluation script
python arc_evaluator.py --checkpoint checkpoints/epoch_12_best.pth
```

## Development Status

### ✅ Week 1: Foundation (Completed)
- [x] Directory structure
- [x] Configuration system (`config.py`)
- [x] Parameter classification (`qwen_param_mapper.py`)
- [x] ARC task adapter (`arc_task.py`)
- [ ] Component testing

### 🚧 Week 2: Bridge & Noise (In Progress)
- [ ] PyTorch-JAX bridge (`pytorch_jax_bridge.py`)
- [ ] EGGROLL noiser integration
- [ ] Noise generation testing
- [ ] Conversion verification

### ⏳ Week 3: Evolution Loop (Pending)
- [ ] Main training script (`arc_evolution_single_gpu.py`)
- [ ] Fitness evaluator (`arc_evaluator.py`)
- [ ] Component integration
- [ ] Smoke test (1 epoch, population=4)

### ⏳ Week 4: Optimization & Validation (Pending)
- [ ] Memory optimization (offloading if needed)
- [ ] Hyperparameter tuning
- [ ] Full 12-epoch training
- [ ] Baseline comparison
- [ ] Results report

## Key Design Decisions

### 1. Single-GPU Architecture

**Challenge:** ERO uses 4 GPUs with Ray distributed workers. EGGROLL uses multi-GPU JAX.

**Solution:** Sequential evaluation with aggressive memory optimization:
- Population size: 1000 → 32-128
- Low-rank perturbations for memory efficiency
- Gradient checkpointing + mixed precision
- Optional model offloading

### 2. Hybrid PyTorch-JAX

**Challenge:** Qwen2.5-VL is PyTorch, EGGROLL is JAX.

**Solution:**
- Keep model in PyTorch (avoid complex conversion)
- Use EGGROLL's JAX noise generation
- Bridge via NumPy arrays

**Alternative:** Pure PyTorch EGGROLL (fallback if bridge too complex)

### 3. Vision Encoder Freezing

**Decision:** Freeze vision encoder by default.

**Rationale:**
- Pretrained encoder already good at visual representations
- Saves ~2-3GB memory
- Improves training stability
- Can unfreeze as experiment

## Performance Targets

| Metric | ERO Baseline | EGGROLL Target |
|--------|--------------|----------------|
| Memory (peak) | 256GB (4×64GB) | 24-48GB (1 GPU) |
| Population size | 1000 | 32-128 |
| Memory/sample | ~256MB | ~2-4MB |
| Final fitness | 1.0 (max) | ≥0.8 (80% of ERO) |

**Success Criteria:**
- ✅ Implementation completes without major blockers
- ✅ Training fits in single GPU memory
- ✅ Fitness improves over epochs
- ✅ Final fitness ≥ 80% of ERO baseline

## Troubleshooting

### Out of Memory

1. Reduce `POPULATION_SIZE` to 16 or 32
2. Enable `MODEL_OFFLOADING = True` in config
3. Reduce `MAX_NEW_TOKENS` to 300
4. Check `FREEZE_VISION = True` and `GRADIENT_CHECKPOINTING = True`
5. Consider 8-bit quantization (requires bitsandbytes)

### Slow Evaluation

1. Ensure CUDA is properly configured
2. Check that model is on GPU: `model.device`
3. Enable torch.compile (if PyTorch 2.0+)
4. Reduce number of ARC tasks for testing

### JAX-PyTorch Conversion Issues

1. Check NumPy intermediate conversion
2. Verify dtype consistency (bfloat16)
3. Test with small tensors first
4. Consider pure PyTorch fallback

## References

- **ERO Paper:** "Evolutionary System 2 Reasoning: An Empirical Proof" (arXiv:2512.05760)
- **EGGROLL Framework:** `HyperscaleES/` directory
- **Qwen2.5-VL:** https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **ARC Challenge:** https://github.com/fchollet/ARC

## Contributing

This is a research project. Key areas for improvement:

1. **Memory optimization:** Better model offloading strategies
2. **Noise generation:** Optimized low-rank sampling
3. **Evaluation:** Faster inference for sequential evaluation
4. **Fitness function:** Alternative metrics beyond character similarity

## License

Follows the licenses of ERO and HyperscaleES components.

## Contact

For questions or issues, please refer to the main project documentation or create an issue in the repository.

---

**Status:** Week 1 (Foundation) - In Progress
**Last Updated:** 2026-01-18

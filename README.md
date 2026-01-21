# Telegrapher's Generative Model via Kac Flows

Official implementation of the paper "Telegrapher's Generative Model via Kac Flows" ([arXiv:2506.20641](https://arxiv.org/abs/2506.20641)).

This repository contains code for the experiments in Section 7, including CIFAR-10 image generation with the Telegrapher/Kac process and baseline comparisons (Flow Matching, Diffusion).

For questions or bug reports contact chemseddine@math.tu-berlin.de.

## Installation

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Repository Structure

```
telegraphers/
├── train_telegraphers.py    # Train Telegrapher model on CIFAR-10
├── train_fm.py              # Train Flow Matching baseline
├── train_diffusion.py       # Train Diffusion baseline
├── train_telegraphers_2d.py # 2D GMM experiment (Telegrapher)
├── train_diffusion_2d.py    # 2D GMM experiment (Diffusion)
├── eval_unified.py          # Unified evaluation script (FID computation)
├── requirements.txt
└── utils/
    ├── unet_oai.py          # UNet architecture (from OpenAI)
    ├── sample_kac.py        # Kac process sampler
    ├── velo_utils.py        # Velocity field computation
    ├── mlp.py               # MLP for 2D experiments
    └── ...
```

## Usage

### Training on CIFAR-10

**Telegrapher Model (Main Method):**
```bash
python train_telegraphers.py \
    --a 9.0 --c 3.0 --T 1.0 \
    --schedule linear \
    --epochs 400000 \
    --batch_size 128 \
    --save_dir results_kac_cifar10 \
    --use_wandb  # optional
```

**Flow Matching Baseline:**
```bash
python train_fm.py \
    --sigma 1.0 \
    --T 1.0 \
    --epochs 400000 \
    --save_dir results_fm_cifar10
```

**Diffusion Baseline:**
```bash
python train_diffusion.py \
    --sigma 1.0 --T 1.0 \
    --schedule linear \
    --epochs 400000 \
    --save_dir results_diff_cifar10
```

### Evaluation

**Evaluate Telegrapher Model:**
```bash
python eval_unified.py \
    --model_type kac \
    --checkpoint_path results_kac_cifar10/ema_400000.pt \
    --a 9.0 --c 3.0 --T 1.0 \
    --num_samples 50000 \
    --compute_nn  # optional: L2 nearest neighbor analysis
```

**Evaluate Flow Matching:**
```bash
python eval_unified.py \
    --model_type fm \
    --checkpoint_path results_fm_cifar10/ema_400000.pt \
    --sigma 1.0 --T 1.0
```

**Evaluate Diffusion:**
```bash
python eval_unified.py \
    --model_type diffusion \
    --checkpoint_path results_diff_cifar10/ema_400000.pt \
    --sigma 1.0 --T 1.0
```

### 2D Experiments

```bash
# Telegrapher on 2D GMM
python train_telegraphers_2d.py --a 9.0 --c 3.0 --T 1.0

# Diffusion on 2D GMM
python train_diffusion_2d.py --sigma 1.0 --T 1.0
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--a` | Kac jump rate | 9.0 |
| `--c` | Kac speed | 3.0 |
| `--T` | Time horizon | 1.0 |
| `--sigma` | Noise scale (diffusion/FM) | 1.0 |
| `--schedule` | Signal schedule (linear/exp/cos/lambda) | linear |
| `--g_schedule` | Noise time reparameterization (t/t2) | t |

## References

[1] R. Duong, J. Chemseddine, P. K. Friz, G. Steidl.
Telegrapher's Generative Model via Kac Flows.

## Citation

```bibtex
@article{DCFS2025,
      title={Telegrapher's Generative Model via Kac Flows},
      author={Richard Duong, Jannis Chemseddine, Peter K. Friz and Gabriele Steidl},
      journal={arXiv preprint arXiv:2506.20641}
}
```

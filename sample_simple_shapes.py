"""
Sample / reconstruct from a trained VAE on simple-shapes-16x16 (RGB).

Usage:
    cd /Users/admin/workspace/torch-vae
    python sample_simple_shapes.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from train_simple_shapes import VAE, SimpleShapesDataset

# ──────────────────────────────────────────────
# Config — must match training
# ──────────────────────────────────────────────
MODEL_PATH  = './models/simple-shapes/vae_simple_shapes_final.pth'
DATASET_DIR = '/Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16'
OUT_DIR     = './figures/simple-shapes-vae'
LATENT_DIMS = 64
C           = 128
NUM_HEADS   = 4
N_SAMPLES   = 64


def to_pil(tensor):
    """(3, H, W) float tensor [0,1] → PIL RGB image."""
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cpu')

    # ── Load model ──
    model = VAE.load(MODEL_PATH, latent_dims=LATENT_DIMS, c=C, num_heads=NUM_HEADS, device=device)
    print(f'Loaded model from {MODEL_PATH}')

    # ── 1. Random samples from prior z ~ N(0, I) ──
    samples_dir = os.path.join(OUT_DIR, 'random_samples')
    os.makedirs(samples_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(N_SAMPLES, LATENT_DIMS, device=device)
        generated = model.decode(z)   # (N, 3, 16, 16)

    for i, img_t in enumerate(generated):
        to_pil(img_t).save(os.path.join(samples_dir, f'{i:04d}.png'))
    print(f'Random samples ({N_SAMPLES}) -> {samples_dir}/')

    # ── 2. Reconstruct real images ──
    recon_dir = os.path.join(OUT_DIR, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)

    dataset = SimpleShapesDataset(DATASET_DIR)
    n_recon = 32
    orig_batch = torch.stack([dataset[i][0] for i in range(n_recon)]).to(device)

    with torch.no_grad():
        recon_batch = model(orig_batch)   # (N, 3, 16, 16)

    for i in range(n_recon):
        to_pil(orig_batch[i]).save(os.path.join(recon_dir, f'{i:04d}_orig.png'))
        to_pil(recon_batch[i]).save(os.path.join(recon_dir, f'{i:04d}_recon.png'))
    print(f'Reconstructions ({n_recon} pairs) -> {recon_dir}/')

    # ── 3. Interpolate between 2 random latent points ──
    interp_dir = os.path.join(OUT_DIR, 'interpolations')
    os.makedirs(interp_dir, exist_ok=True)

    n_steps = 10
    with torch.no_grad():
        z_a = torch.randn(1, LATENT_DIMS, device=device)
        z_b = torch.randn(1, LATENT_DIMS, device=device)
        for step, t in enumerate(np.linspace(0, 1, n_steps)):
            z_interp = (1 - t) * z_a + t * z_b
            img_t = model.decode(z_interp).squeeze(0)   # (3, 16, 16)
            to_pil(img_t).save(os.path.join(interp_dir, f'{step:02d}_t{t:.2f}.png'))
    print(f'Interpolation ({n_steps} steps) -> {interp_dir}/')

    print('\nDone.')

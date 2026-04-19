"""
Sample / reconstruct from a trained VAE on simple-shapes-16x16.

Usage:
    cd /Users/admin/workspace/torch-vae
    python sample_simple_shapes.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from train_simple_shapes import VAE, SimpleShapesDataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH  = './models/simple-shapes/vae_simple_shapes_final.pth'
DATASET_DIR = '/Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16'
OUT_DIR     = './figures/simple-shapes-vae'
LATENT_DIMS = 32
NUM_FILTERS = 64
N_SAMPLES   = 64


def save_img(arr, path):
    """arr: numpy (H, W) float [0,1]"""
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8), mode='L').save(path)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cpu')

    # ── Load model ──
    model = VAE.load(MODEL_PATH, latent_dims=LATENT_DIMS, num_filters=NUM_FILTERS, device=device)
    print(f'Loaded model from {MODEL_PATH}')

    # ── 1. Random samples from prior z ~ N(0, I) ──
    samples_dir = os.path.join(OUT_DIR, 'random_samples')
    os.makedirs(samples_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(N_SAMPLES, LATENT_DIMS)
        generated = model.decode(z)   # (N, 1, 16, 16)

    for i, img in enumerate(generated):
        save_img(img.squeeze(0).numpy(), os.path.join(samples_dir, f'{i:04d}.png'))
    print(f'Random samples ({N_SAMPLES}) -> {samples_dir}/')

    # ── 2. Reconstruct real images ──
    recon_dir = os.path.join(OUT_DIR, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)

    dataset = SimpleShapesDataset(DATASET_DIR)
    n_recon = 32
    orig_batch = torch.stack([dataset[i][0] for i in range(n_recon)])  # (N, 1, 16, 16)

    with torch.no_grad():
        recon_batch = model(orig_batch)   # (N, 1, 16, 16)

    for i in range(n_recon):
        save_img(orig_batch[i].squeeze(0).numpy(),  os.path.join(recon_dir, f'{i:04d}_orig.png'))
        save_img(recon_batch[i].squeeze(0).numpy(), os.path.join(recon_dir, f'{i:04d}_recon.png'))
    print(f'Reconstructions ({n_recon} pairs) -> {recon_dir}/')

    # ── 3. Interpolate between 2 random latent points ──
    interp_dir = os.path.join(OUT_DIR, 'interpolations')
    os.makedirs(interp_dir, exist_ok=True)

    n_steps = 10
    with torch.no_grad():
        z_a = torch.randn(1, LATENT_DIMS)
        z_b = torch.randn(1, LATENT_DIMS)
        for step, t in enumerate(np.linspace(0, 1, n_steps)):
            z_interp = (1 - t) * z_a + t * z_b
            img = model.decode(z_interp).squeeze().numpy()
            save_img(img, os.path.join(interp_dir, f'{step:02d}_t{t:.2f}.png'))
    print(f'Interpolation ({n_steps} steps) -> {interp_dir}/')

    print('\nDone.')

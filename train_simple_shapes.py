"""
Train a VAE on simple-shapes-16x16 (grayscale, 5000 images).
Runs on CPU (macOS) — no GPU needed at this scale.

Architecture: deeper 3-layer CNN VAE for 16x16 input.
  Encoder: 16x16 --(s1)--> 16x16 --(s2)--> 8x8 --(s2)--> 4x4 --(s2)--> 2x2 -> mu/logvar
  Decoder: z -> 2x2 --(s2)--> 4x4 --(s2)--> 8x8 --(s2)--> 16x16 -> sigmoid

Key improvements over v1:
  - 3x more filters (64/128/256)
  - Extra conv at full resolution before downsampling
  - BCE loss instead of MSE -> sharper binary-ish images
  - KL weight = 0.001 (beta-VAE style) -> prioritize reconstruction sharpness
  - Latent dim = 32
  - 300 epochs

Usage:
    cd /Users/admin/workspace/torch-vae
    python train_simple_shapes.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Trainer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_DIR = '/Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16'
MODEL_DIR   = './models/simple-shapes'
LATENT_DIMS = 32
NUM_FILTERS = 64     # base width; layers use F, 2F, 4F channels
BATCH_SIZE  = 256    # larger batch on GPU
EPOCHS      = 300
LR          = 3e-4
KL_WEIGHT   = 0.001  # beta-VAE: low beta -> reconstruction-focused -> sharper outputs


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class SimpleShapesDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        )
        self.transform = transforms.ToTensor()   # [0,1], shape (1, H, W)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img), 0


# ──────────────────────────────────────────────
# Loss — BCE + KL  (better than MSE for binary-ish images)
# ──────────────────────────────────────────────
def bce_kl_loss(x, x_recon, kl_div, beta=KL_WEIGHT):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    return recon_loss + beta * kl_div


# ──────────────────────────────────────────────
# Model — deeper CNN VAE for 16x16 grayscale
# ──────────────────────────────────────────────
class ResBlock(nn.Module):
    """Lightweight residual block — same spatial size, same channels."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, latent_dims, f):
        super().__init__()
        # (1, 16, 16)
        self.net = nn.Sequential(
            # stage 0: refine at full resolution
            nn.Conv2d(1,   f,   3, stride=1, padding=1, bias=False), nn.BatchNorm2d(f),   nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f),
            # stage 1: 16x16 -> 8x8
            nn.Conv2d(f,   f*2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(f*2), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f*2),
            # stage 2: 8x8 -> 4x4
            nn.Conv2d(f*2, f*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(f*4), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f*4),
            # stage 3: 4x4 -> 2x2
            nn.Conv2d(f*4, f*4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(f*4), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = f * 4 * 2 * 2
        self.mu      = nn.Linear(self.flat_dim, latent_dims)
        self.log_var = nn.Linear(self.flat_dim, latent_dims)

    def forward(self, x):
        h = self.net(x).flatten(1)
        mu      = self.mu(h)
        log_var = self.log_var(h)
        eps     = torch.randn_like(mu)
        z       = mu + eps * torch.exp(0.5 * log_var)
        kl_div  = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl_div


class Decoder(nn.Module):
    def __init__(self, latent_dims, f):
        super().__init__()
        self.flat_dim = f * 4 * 2 * 2
        self.f4 = f * 4
        self.fc = nn.Linear(latent_dims, self.flat_dim)
        # (f*4, 2, 2)
        self.net = nn.Sequential(
            # stage 3: 2x2 -> 4x4
            nn.ConvTranspose2d(f*4, f*4, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(f*4), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f*4),
            # stage 2: 4x4 -> 8x8
            nn.ConvTranspose2d(f*4, f*2, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(f*2), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f*2),
            # stage 1: 8x8 -> 16x16
            nn.ConvTranspose2d(f*2, f,   4, stride=2, padding=1, bias=False), nn.BatchNorm2d(f),   nn.LeakyReLU(0.2, inplace=True),
            ResBlock(f),
            # output: refine + sigmoid
            nn.Conv2d(f, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.f4, 2, 2)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dims=32, num_filters=64):
        super().__init__()
        self.encoder = Encoder(latent_dims, num_filters)
        self.decoder = Decoder(latent_dims, num_filters)
        self.kl_div  = torch.tensor(0.0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, self.kl_div = self.encoder(x)
        return self.decoder(z)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f'Model saved -> {path}')

    @staticmethod
    def load(path, latent_dims=32, num_filters=64, device='cpu'):
        model = VAE(latent_dims, num_filters)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    dataset = SimpleShapesDataset(DATASET_DIR)
    print(f'Dataset: {len(dataset)} images')

    use_cuda = device.type == 'cuda'
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = VAE(latent_dims=LATENT_DIMS, num_filters=NUM_FILTERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = lambda x, x_recon, kl: bce_kl_loss(x, x_recon, kl, beta=KL_WEIGHT)

    save_prefix = os.path.join(MODEL_DIR, 'vae_simple_shapes')

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        fname_save_every_epoch=save_prefix,
    )
    trainer.train(loader, None, num_epochs=EPOCHS)

    final_path = os.path.join(MODEL_DIR, 'vae_simple_shapes_final.pth')
    model.save(final_path)
    print(f'\nDone. Final model -> {final_path}')

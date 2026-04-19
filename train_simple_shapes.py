"""
Train a VAE on simple-shapes-16x16 (RGB 3-channel).

Architecture:
  Encoder/Decoder backbone = UNet components from improved-diffusion
  (ResBlock + AttentionBlock + Downsample/Upsample)
  No skip connections — pure encoder / decoder bottleneck.

  Encoder: (3,16,16) → C→2C→4C with downsampling → flatten → mu, logvar
  Decoder: latent → reshape(4C,2,2) → upsample to (3,16,16)

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

# ── import UNet building blocks from improved-diffusion ──
sys.path.insert(0, '/Users/admin/workspace/improved-diffusion')
from improved_diffusion.unet import ResBlock, AttentionBlock, Downsample, Upsample
from improved_diffusion.nn import normalization, zero_module

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Trainer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_DIR = '/Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16'
MODEL_DIR   = './models/simple-shapes'
LATENT_DIMS = 64
C           = 128    # base channel width; encoder uses C → 2C → 4C
DROPOUT     = 0.0
NUM_HEADS   = 4
BATCH_SIZE  = 256
EPOCHS      = 300
LR          = 2e-4
KL_WEIGHT_START  = 0.0001  # start very low → decoder learns to reconstruct first
KL_WEIGHT_END    = 0.5     # end value → posterior converges toward N(0,1)
KL_ANNEAL_EPOCHS = 200     # linearly anneal over first N epochs


# ──────────────────────────────────────────────
# Dataset — RGB
# ──────────────────────────────────────────────
class SimpleShapesDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        )
        self.transform = transforms.ToTensor()   # (3, H, W), [0, 1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), 0


# ──────────────────────────────────────────────
# Helper: ResBlock wrapper — no timestep needed for VAE
# ──────────────────────────────────────────────
class Res(nn.Module):
    """Wrap improved-diffusion ResBlock; passes a fixed zero embedding."""
    def __init__(self, in_ch, out_ch=None, dropout=0.0, use_scale_shift_norm=True):
        super().__init__()
        out_ch = out_ch or in_ch
        emb_ch = in_ch * 4   # standard ratio in improved-diffusion
        self.block = ResBlock(
            channels=in_ch,
            emb_channels=emb_ch,
            dropout=dropout,
            out_channels=out_ch,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=2,
        )
        self._emb_ch = emb_ch

    def forward(self, x):
        emb = torch.zeros(x.shape[0], self._emb_ch, device=x.device, dtype=x.dtype)
        return self.block(x, emb)


def Attn(channels, num_heads=NUM_HEADS):
    return AttentionBlock(channels, num_heads=num_heads)


# ──────────────────────────────────────────────
# Encoder  (3,16,16) → (LATENT_DIMS,)  x2 for mu/logvar
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dims, c, dropout, num_heads):
        super().__init__()
        # (3, 16, 16) → (C, 16, 16)
        self.stem = nn.Conv2d(3, c, 3, padding=1)

        # 16x16 → 8x8
        self.down1 = nn.Sequential(
            Res(c, c, dropout),
            Attn(c, num_heads),
            Downsample(c, use_conv=True),
        )

        # 8x8 → 4x4
        self.down2 = nn.Sequential(
            Res(c, c * 2, dropout),
            Attn(c * 2, num_heads),
            Downsample(c * 2, use_conv=True),
        )

        # 4x4 → 2x2
        self.down3 = nn.Sequential(
            Res(c * 2, c * 4, dropout),
            Attn(c * 4, num_heads),
            Downsample(c * 4, use_conv=True),
        )

        # bottleneck at 2x2
        self.mid = nn.Sequential(
            Res(c * 4, c * 4, dropout),
            Attn(c * 4, num_heads),
            Res(c * 4, c * 4, dropout),
        )

        flat_dim = c * 4 * 2 * 2
        self.norm_out = normalization(c * 4)
        self.act      = nn.SiLU()
        self.fc_mu    = nn.Linear(flat_dim, latent_dims)
        self.fc_lv    = nn.Linear(flat_dim, latent_dims)

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.act(self.norm_out(h)).flatten(1)
        mu      = self.fc_mu(h)
        log_var = self.fc_lv(h)
        eps = torch.randn_like(mu)
        z   = mu + eps * torch.exp(0.5 * log_var)
        kl  = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl


# ──────────────────────────────────────────────
# Decoder  (LATENT_DIMS,) → (3,16,16)
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dims, c, dropout, num_heads):
        super().__init__()
        flat_dim = c * 4 * 2 * 2
        self.c4  = c * 4
        self.fc  = nn.Linear(latent_dims, flat_dim)

        # bottleneck at 2x2
        self.mid = nn.Sequential(
            Res(c * 4, c * 4, dropout),
            Attn(c * 4, num_heads),
            Res(c * 4, c * 4, dropout),
        )

        # 2x2 → 4x4
        self.up3 = nn.Sequential(
            Upsample(c * 4, use_conv=True),
            Res(c * 4, c * 2, dropout),
            Attn(c * 2, num_heads),
        )

        # 4x4 → 8x8
        self.up2 = nn.Sequential(
            Upsample(c * 2, use_conv=True),
            Res(c * 2, c, dropout),
            Attn(c, num_heads),
        )

        # 8x8 → 16x16
        self.up1 = nn.Sequential(
            Upsample(c, use_conv=True),
            Res(c, c, dropout),
            Attn(c, num_heads),
        )

        self.norm_out = normalization(c)
        self.act      = nn.SiLU()
        self.conv_out = zero_module(nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, z):
        h = self.fc(z).view(-1, self.c4, 2, 2)
        h = self.mid(h)
        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.act(self.norm_out(h))
        return torch.sigmoid(self.conv_out(h))


# ──────────────────────────────────────────────
# VAE
# ──────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dims=64, c=128, dropout=0.0, num_heads=4):
        super().__init__()
        self.encoder = Encoder(latent_dims, c, dropout, num_heads)
        self.decoder = Decoder(latent_dims, c, dropout, num_heads)
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
    def load(path, latent_dims=64, c=128, dropout=0.0, num_heads=4, device='cpu'):
        model = VAE(latent_dims, c, dropout, num_heads)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


# ──────────────────────────────────────────────
# Loss — BCE + KL
# ──────────────────────────────────────────────
def bce_kl_loss(x, x_recon, kl_div, beta):
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    return recon + beta * kl_div


# ──────────────────────────────────────────────
# Trainer — KL annealing + save every N epochs
# ──────────────────────────────────────────────
SAVE_EVERY = 50   # checkpoint every N epochs

class AnnealingTrainer(Trainer):
    """Linearly anneals KL weight from KL_WEIGHT_START to KL_WEIGHT_END
    over the first KL_ANNEAL_EPOCHS epochs, then keeps it at KL_WEIGHT_END."""

    def _get_beta(self, epoch):
        if KL_ANNEAL_EPOCHS == 0:
            return KL_WEIGHT_END
        t = min(epoch / KL_ANNEAL_EPOCHS, 1.0)
        return KL_WEIGHT_START + t * (KL_WEIGHT_END - KL_WEIGHT_START)

    def _train_epoch(self, epoch, train_loader):
        beta = self._get_beta(epoch)
        # patch loss_fn with current beta
        self.loss_fn = lambda x, x_recon, kl: bce_kl_loss(x, x_recon, kl, beta=beta)
        return super()._train_epoch(epoch, train_loader)

    def _on_epoch_end(self, epoch, num_train_samples, num_batches):
        import time
        self.metrics["epochEndTime"] = time.monotonic()
        self._update_metrics(epoch, num_train_samples)
        self._log_metrics(epoch)
        beta = self._get_beta(epoch)
        self.logger.info(f'  KL beta: {beta:.5f}\n')
        if (epoch + 1) % SAVE_EVERY == 0:
            self._save_model(self.fname_save_every_epoch, epoch)


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
    print(f'Dataset: {len(dataset)} images (RGB)')

    use_cuda = device.type == 'cuda'
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = VAE(latent_dims=LATENT_DIMS, c=C, dropout=DROPOUT, num_heads=NUM_HEADS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # placeholder — AnnealingTrainer overrides loss_fn each epoch with current beta
    initial_loss_fn = lambda x, x_recon, kl: bce_kl_loss(x, x_recon, kl, beta=KL_WEIGHT_START)

    trainer = AnnealingTrainer(
        model=model,
        loss_fn=initial_loss_fn,
        optimizer=optimizer,
        device=device,
        fname_save_every_epoch=os.path.join(MODEL_DIR, 'vae_simple_shapes'),
    )
    trainer.train(loader, None, num_epochs=EPOCHS)

    final_path = os.path.join(MODEL_DIR, 'vae_simple_shapes_final.pth')
    model.save(final_path)
    print(f'\nDone. Final model -> {final_path}')

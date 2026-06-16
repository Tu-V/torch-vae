"""
VQ-VAE for simple-shapes-16x16 (quality-focused redesign).

Key differences from the first attempt:
  1. Spatial 4×4 latent map (16 code slots per image) instead of a single
     vector. Spatial structure is preserved — the decoder gets local context
     about what belongs where, instead of trying to unpack everything from
     one point.

  2. EMA codebook updates (van den Oord et al. 2017, Appendix A.1) instead
     of gradient-based codebook loss. EMA is far more stable and the main
     reason the original paper avoided code collapse in practice.

  3. Dead-code restart: any code unused in the current batch is re-initialised
     to a random encoder output from that batch. This keeps the full codebook
     alive throughout training.

  4. Small, clean architecture (~540K params, no external deps).
     Appropriate for 5K images at 16×16.

Latent space:
  - Each image → encoder → (D, 4, 4) continuous map
  - Each of the 16 spatial positions is snapped to the nearest of K codes
  - Decoder reconstructs from the 4×4 grid of quantised vectors

Generation (see sample_simple_shapes_vqvae.py):
  - Find active codes (used by ≥1 training image at any position)
  - For each of 16 positions, independently sample one active code
  - Decode the resulting (4, 4) code grid

Usage:
    cd /Users/admin/workspace/diffusion_hallu/torch-vae
    python train_simple_shapes_vqvae.py
"""

import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_simple_shapes import SimpleShapesDataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_DIR = ('/Users/admin/workspace/diffusion_hallu/neurips-2024-diffusion-model-hallucination'
               '/simple-datasets/simple-shapes-5k-16x16')
MODEL_DIR   = './models/simple-shapes-5k-16x16-vqvae'

K          = 128    # codebook size
D          = 32     # codebook / encoder output dimension
C          = 64     # base channel width
BETA       = 0.25   # commitment loss weight
EMA_DECAY  = 0.99   # EMA smoothing for codebook updates

LATENT_H   = LATENT_W = 4   # encoder output spatial size (16 → 4)

BATCH_SIZE = 256
EPOCHS     = 500
LR         = 2e-4
SAVE_EVERY = 50
TOP_K_BEST = 10


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────
class Res(nn.Module):
    """Residual block with GroupNorm + SiLU."""
    def __init__(self, C: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


# ──────────────────────────────────────────────
# Encoder  (3,16,16) → (D,4,4)
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, D: int = D, C: int = C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 4, stride=2, padding=1),   # 16→8
            Res(C),
            nn.Conv2d(C, C, 4, stride=2, padding=1),   # 8→4
            Res(C),
            nn.Conv2d(C, D, 1),                         # project to codebook dim
        )

    def forward(self, x):
        return self.net(x)   # (B, D, 4, 4)


# ──────────────────────────────────────────────
# Decoder  (D,4,4) → (3,16,16)
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, D: int = D, C: int = C):
        super().__init__()
        self.proj = nn.Conv2d(D, C, 3, padding=1)
        self.res1 = Res(C)
        self.up1  = nn.Sequential(                      # 4→8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(C, C, 3, padding=1),
        )
        self.res2 = Res(C)
        self.up2  = nn.Sequential(                      # 8→16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(C, C, 3, padding=1),
        )
        self.res3 = Res(C)
        self.out  = nn.Conv2d(C, 3, 1)

    def forward(self, z_q):
        h = self.proj(z_q)
        h = self.res1(h)
        h = self.up1(h)
        h = self.res2(h)
        h = self.up2(h)
        h = self.res3(h)
        return torch.sigmoid(self.out(h))


# ──────────────────────────────────────────────
# Vector Quantisation — EMA codebook
# ──────────────────────────────────────────────
class VQLayer(nn.Module):
    """
    VQ with EMA codebook updates (no gradient to codebook).

    During forward (training mode):
      - Assign each spatial encoder output to its nearest code
      - Update codebook via EMA: each code drifts toward the mean of
        encoder outputs assigned to it in this batch
      - Dead-code restart: codes unused this batch are re-initialised
        to a random encoder output from the same batch
    """
    def __init__(self, K: int = K, D: int = D,
                 beta: float = BETA, decay: float = EMA_DECAY, eps: float = 1e-5):
        super().__init__()
        self.K     = K
        self.beta  = beta
        self.decay = decay
        self.eps   = eps

        embed = torch.randn(K, D) * 0.1
        self.register_buffer('embedding',  embed)
        self.register_buffer('ema_count',  torch.ones(K) * 1e-4)
        self.register_buffer('ema_weight', embed.clone())

    def forward(self, z_e: torch.Tensor):
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)   # (BHW, D)

        # Squared L2 distance to every codebook entry
        dist = (flat.pow(2).sum(1, keepdim=True)
                + self.embedding.pow(2).sum(1)
                - 2 * flat @ self.embedding.t())        # (BHW, K)
        idx_flat = dist.argmin(dim=1)                   # (BHW,)
        z_q_flat = self.embedding[idx_flat]             # (BHW, D)

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(idx_flat, self.K).float()   # (BHW, K)
                count   = one_hot.sum(0)                          # (K,)

                # EMA update
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_weight.mul_(self.decay).add_(
                    one_hot.t() @ flat, alpha=1 - self.decay)

                # Laplace-smoothed mean per code
                n      = self.ema_count.sum()
                smooth = ((self.ema_count + self.eps)
                          / (n + self.K * self.eps) * n)
                self.embedding.copy_(self.ema_weight / smooth.unsqueeze(1))

                # Dead-code restart
                dead   = (count == 0)
                n_dead = int(dead.sum())
                if n_dead > 0:
                    ri = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
                    self.embedding[dead] = flat[ri].detach()

        z_q    = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        commit = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()                      # straight-through
        return z_q_st, commit, idx_flat.view(B, H, W)


# ──────────────────────────────────────────────
# VQ-VAE
# ──────────────────────────────────────────────
class VQVAE(nn.Module):
    LATENT_H = LATENT_W = 4

    def __init__(self, K: int = K, D: int = D, C: int = C, beta: float = BETA):
        super().__init__()
        self.encoder = Encoder(D, C)
        self.vq      = VQLayer(K, D, beta)
        self.decoder = Decoder(D, C)

    def forward(self, x):
        z_e             = self.encoder(x)
        z_q_st, commit, idx = self.vq(z_e)
        x_recon         = self.decoder(z_q_st)
        return x_recon, commit, idx

    def decode_codes(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, H, W) long tensor of code indices
        Returns (B, 3, 16, 16) decoded images.
        """
        B, H, W = idx.shape
        z_q = (self.vq.embedding[idx.reshape(-1)]
                    .view(B, H, W, -1)
                    .permute(0, 3, 1, 2))   # (B, D, H, W)
        return self.decoder(z_q)

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x → code index map (B, H, W). No grad."""
        with torch.no_grad():
            _, _, idx = self(x)
        return idx

    def active_codes(self, loader, device) -> list:
        """Return sorted list of code indices used by ≥1 training image."""
        used = set()
        self.eval()
        with torch.no_grad():
            for imgs, _ in loader:
                _, _, idx = self(imgs.to(device))
                used.update(idx.reshape(-1).tolist())
        return sorted(used)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str, K: int = K, D: int = D, C: int = C,
             beta: float = BETA, device: str = 'cpu') -> 'VQVAE':
        m = VQVAE(K, D, C, beta)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = (torch.device('cuda')  if torch.cuda.is_available()  else
              torch.device('mps')   if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f'Device : {device}')
    print(f'Config : K={K}, D={D}, C={C}, beta={BETA}, ema_decay={EMA_DECAY}')
    print(f'Latent : {LATENT_H}×{LATENT_W} spatial map = {LATENT_H*LATENT_W} codes/image')

    dataset = SimpleShapesDataset(DATASET_DIR)
    print(f'Dataset: {len(dataset)} images  ({DATASET_DIR})')
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model   = VQVAE(K=K, D=D, C=C, beta=BETA).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params : {n_params:,}')

    # Codebook is a buffer (EMA), only encoder+decoder go into optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loss_log = os.path.join(MODEL_DIR, 'vqvae_loss_log.csv')
    with open(loss_log, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'loss', 'recon_loss', 'commit_loss', 'active_codes', 'time_sec'])

    best_loss        = float('inf')
    best_checkpoints = []   # [(loss, epoch, path)] sorted asc by loss

    for epoch in range(EPOCHS):
        t0 = time.monotonic()
        model.train()
        sum_loss = sum_recon = sum_commit = 0.0
        code_hits = torch.zeros(K, device=device)

        for imgs, _ in loader:
            imgs = imgs.to(device)
            x_recon, commit, idx = model(imgs)

            recon_loss = F.binary_cross_entropy(x_recon, imgs, reduction='sum') / imgs.shape[0]
            loss       = recon_loss + BETA * commit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss   += loss.item()
            sum_recon  += recon_loss.item()
            sum_commit += commit.item()
            code_hits.scatter_add_(0, idx.reshape(-1),
                                   torch.ones(idx.numel(), device=device))

        n          = len(loader)
        epoch_loss = sum_loss   / n
        recon_avg  = sum_recon  / n
        commit_avg = sum_commit / n
        n_active   = int((code_hits > 0).sum())
        elapsed    = time.monotonic() - t0

        print(f'epoch {epoch:3d} | loss={epoch_loss:8.2f}  '
              f'recon={recon_avg:8.2f}  commit={commit_avg:.4f}  '
              f'active={n_active:3d}/{K}  ({elapsed:.1f}s)')

        with open(loss_log, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{epoch_loss:.6f}', f'{recon_avg:.6f}',
                f'{commit_avg:.6f}', n_active, f'{elapsed:.3f}',
            ])

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.save(os.path.join(MODEL_DIR, 'vqvae_best.pth'))
            print(f'  *** new best {epoch_loss:.2f} at epoch {epoch}')

        if len(best_checkpoints) < TOP_K_BEST or epoch_loss < best_checkpoints[-1][0]:
            p = os.path.join(MODEL_DIR, f'vqvae_top_ep{epoch:03d}_loss{epoch_loss:.2f}.pth')
            model.save(p)
            best_checkpoints.append((epoch_loss, epoch, p))
            best_checkpoints.sort(key=lambda x: x[0])
            if len(best_checkpoints) > TOP_K_BEST:
                _, _, old = best_checkpoints.pop()
                if os.path.exists(old):
                    os.remove(old)

        if (epoch + 1) % SAVE_EVERY == 0:
            model.save(os.path.join(MODEL_DIR, f'vqvae_ep{epoch:03d}.pth'))

    print(f'\nDone. Best loss: {best_loss:.4f}')
    print(f'Log        -> {loss_log}')
    print(f'Checkpoint -> {MODEL_DIR}/vqvae_best.pth')

"""
Train a VQ-VAE on simple-shapes-16x16.

Key difference from train_simple_shapes.py (continuous VAE):
  - No mu / log_var, no reparameterization, no KL term
  - Encoder output is "snapped" to the nearest of K codebook vectors
    (vector quantization). The decoder only ever receives one of K
    discrete codes — no smooth interpolation zone between training images
    → eliminates "blending hallucination" caused by the continuous VAE.

Loss = BCE_recon + codebook_loss + beta * commitment_loss

  codebook_loss   : pulls codebook entries toward encoder outputs (sg on z_e)
  commitment_loss : pulls encoder outputs toward nearest codebook entry (sg on z_q)
  Both use straight-through estimator so gradients flow to the encoder.

Sampling at generation time:
  - Only sample code indices that were actually used during training
    (active codes). Dead codes produce garbage — decoder never saw them.
  - See decode_code() and active_codes() helpers below.

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
sys.path.insert(0, '/Users/admin/workspace/diffusion_hallu/improved-diffusion')
from improved_diffusion.unet import ResBlock, AttentionBlock, Downsample, Upsample
from improved_diffusion.nn import normalization, zero_module
from train_simple_shapes import SimpleShapesDataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_DIR = ('/Users/admin/workspace/diffusion_hallu/neurips-2024-diffusion-model-hallucination'
               '/simple-datasets/simple-shapes-5k-16x16')
MODEL_DIR   = './models/simple-shapes-5k-16x16-vqvae'

K           = 512    # codebook size (number of discrete codes)
D           = 64     # codebook entry dimension
BETA        = 0.25   # commitment loss weight
C           = 128    # base channel width (same as VAE)
NUM_HEADS   = 4
DROPOUT     = 0.0
BATCH_SIZE  = 256
EPOCHS      = 300
LR          = 2e-4
SAVE_EVERY  = 50
TOP_K_BEST  = 10


# ──────────────────────────────────────────────
# Building blocks (same as VAE, shared Res wrapper)
# ──────────────────────────────────────────────
class Res(nn.Module):
    """improved-diffusion ResBlock with a fixed zero timestep embedding."""
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch  = out_ch or in_ch
        emb_ch  = in_ch * 4
        self.block = ResBlock(
            channels=in_ch, emb_channels=emb_ch, dropout=dropout,
            out_channels=out_ch, use_scale_shift_norm=True, dims=2,
        )
        self._emb_ch = emb_ch

    def forward(self, x):
        emb = torch.zeros(x.shape[0], self._emb_ch, device=x.device, dtype=x.dtype)
        return self.block(x, emb)


def Attn(ch):
    return AttentionBlock(ch, num_heads=NUM_HEADS)


# ──────────────────────────────────────────────
# Vector Quantization layer
# ──────────────────────────────────────────────
class VectorQuantizer(nn.Module):
    """
    Straight-through VQ.

    Forward:
      z_e  (B, D) continuous encoder output
      -> finds nearest codebook entry for each sample
      -> returns z_q_st (B, D) with straight-through gradient,
         vq_loss scalar, and code indices (B,)

    The straight-through trick:
      z_q_st = z_e + (z_q - z_e).detach()
      gradient flows through z_e unmodified,
      while z_q (with no gradient) shifts the encoder toward the codebook.
    """
    def __init__(self, K, D, beta=0.25):
        super().__init__()
        self.K    = K
        self.D    = D
        self.beta = beta
        self.embedding = nn.Embedding(K, D)
        nn.init.uniform_(self.embedding.weight, -1 / K, 1 / K)

    def forward(self, z_e):
        # pairwise squared L2:  ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 z_e·e_k
        dist = (z_e.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * z_e @ self.embedding.weight.t())   # (B, K)
        idx  = dist.argmin(dim=1)                        # (B,)
        z_q  = self.embedding(idx)                       # (B, D)

        codebook_loss   = F.mse_loss(z_q,  z_e.detach())   # moves codebook → z_e
        commitment_loss = F.mse_loss(z_e,  z_q.detach())   # moves z_e → codebook
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()               # straight-through
        return z_q_st, vq_loss, idx

    def lookup(self, idx):
        return self.embedding(idx)


# ──────────────────────────────────────────────
# Encoder  (3,16,16) → (D,)
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, D, c=C, dropout=DROPOUT):
        super().__init__()
        self.stem  = nn.Conv2d(3, c, 3, padding=1)
        self.down1 = nn.Sequential(Res(c,   c,   dropout), Attn(c),   Downsample(c,   use_conv=True))
        self.down2 = nn.Sequential(Res(c,   c*2, dropout), Attn(c*2), Downsample(c*2, use_conv=True))
        self.down3 = nn.Sequential(Res(c*2, c*4, dropout), Attn(c*4), Downsample(c*4, use_conv=True))
        self.mid   = nn.Sequential(Res(c*4, c*4, dropout), Attn(c*4), Res(c*4, c*4, dropout))
        self.norm_out = normalization(c*4)
        self.act      = nn.SiLU()
        self.fc_e     = nn.Linear(c*4*2*2, D)

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.act(self.norm_out(h)).flatten(1)
        return self.fc_e(h)   # (B, D)


# ──────────────────────────────────────────────
# Decoder  (D,) → (3,16,16)
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, D, c=C, dropout=DROPOUT):
        super().__init__()
        self.c4 = c*4
        self.fc = nn.Linear(D, c*4*2*2)
        self.mid = nn.Sequential(Res(c*4, c*4, dropout), Attn(c*4), Res(c*4, c*4, dropout))
        self.up3 = nn.Sequential(Upsample(c*4, use_conv=True), Res(c*4, c*2, dropout), Attn(c*2))
        self.up2 = nn.Sequential(Upsample(c*2, use_conv=True), Res(c*2, c,   dropout), Attn(c))
        self.up1 = nn.Sequential(Upsample(c,   use_conv=True), Res(c,   c,   dropout), Attn(c))
        self.norm_out = normalization(c)
        self.act      = nn.SiLU()
        self.conv_out = zero_module(nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, z_q):
        h = self.fc(z_q).view(-1, self.c4, 2, 2)
        h = self.mid(h)
        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.act(self.norm_out(h))
        return torch.sigmoid(self.conv_out(h))


# ──────────────────────────────────────────────
# VQ-VAE
# ──────────────────────────────────────────────
class VQVAE(nn.Module):
    def __init__(self, K=K, D=D, c=C, dropout=DROPOUT, beta=BETA):
        super().__init__()
        self.encoder = Encoder(D, c, dropout)
        self.vq      = VectorQuantizer(K, D, beta)
        self.decoder = Decoder(D, c, dropout)

    def forward(self, x):
        z_e             = self.encoder(x)
        z_q_st, vq_loss, idx = self.vq(z_e)
        x_recon         = self.decoder(z_q_st)
        return x_recon, vq_loss, idx

    def decode_code(self, idx):
        """Decode from codebook index (for sampling at generation time)."""
        return self.decoder(self.vq.lookup(idx))

    def active_codes(self, loader, device):
        """Return set of code indices that at least 1 training image maps to."""
        used = set()
        self.eval()
        with torch.no_grad():
            for imgs, _ in loader:
                z_e = self.encoder(imgs.to(device))
                _, _, idx = self.vq(z_e)
                used.update(idx.cpu().tolist())
        return sorted(used)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, K=K, D=D, c=C, dropout=DROPOUT, beta=BETA, device='cpu'):
        m = VQVAE(K, D, c, dropout, beta)
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
    print(f'Device: {device}')
    print(f'Config: K={K} codes, D={D}-dim codes, C={C} channels, beta={BETA}')

    dataset = SimpleShapesDataset(DATASET_DIR)
    print(f'Dataset: {len(dataset)} images  ({DATASET_DIR})')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VQVAE(K=K, D=D, c=C, dropout=DROPOUT, beta=BETA).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loss_log = os.path.join(MODEL_DIR, 'vqvae_loss_log.csv')
    with open(loss_log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'recon_loss', 'vq_loss', 'active_codes', 'time_sec'])

    best_loss        = float('inf')
    best_epoch       = -1
    best_checkpoints = []   # [(loss, epoch, path)] sorted asc by loss

    for epoch in range(EPOCHS):
        t0 = time.monotonic()
        model.train()
        sum_loss = sum_recon = sum_vq = 0.0
        code_hits = torch.zeros(K, device=device)

        for imgs, _ in loader:
            imgs = imgs.to(device)
            x_recon, vq_loss, idx = model(imgs)

            recon_loss = F.binary_cross_entropy(x_recon, imgs, reduction='sum') / imgs.shape[0]
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss  += loss.item()
            sum_recon += recon_loss.item()
            sum_vq    += vq_loss.item()
            code_hits.scatter_add_(0, idx, torch.ones(len(idx), device=device))

        n          = len(loader)
        epoch_loss = sum_loss  / n
        recon_avg  = sum_recon / n
        vq_avg     = sum_vq    / n
        n_active   = int((code_hits > 0).sum())
        elapsed    = time.monotonic() - t0

        print(f'epoch {epoch:3d} | loss={epoch_loss:8.2f}  '
              f'recon={recon_avg:8.2f}  vq={vq_avg:.4f}  '
              f'active_codes={n_active:3d}/{K}  ({elapsed:.1f}s)')

        with open(loss_log, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{epoch_loss:.6f}', f'{recon_avg:.6f}',
                f'{vq_avg:.6f}', n_active, f'{elapsed:.3f}',
            ])

        # best overall
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            model.save(os.path.join(MODEL_DIR, 'vqvae_best.pth'))
            print(f'  *** new best {epoch_loss:.2f} at epoch {epoch}')

        # top-K by loss
        if len(best_checkpoints) < TOP_K_BEST or epoch_loss < best_checkpoints[-1][0]:
            p = os.path.join(MODEL_DIR, f'vqvae_top_ep{epoch:03d}_loss{epoch_loss:.2f}.pth')
            model.save(p)
            best_checkpoints.append((epoch_loss, epoch, p))
            best_checkpoints.sort(key=lambda x: x[0])
            if len(best_checkpoints) > TOP_K_BEST:
                _, _, old = best_checkpoints.pop()
                if os.path.exists(old):
                    os.remove(old)

        # periodic
        if (epoch + 1) % SAVE_EVERY == 0:
            model.save(os.path.join(MODEL_DIR, f'vqvae_{epoch:03d}.pth'))

    print(f'\nDone. Best: epoch={best_epoch}, loss={best_loss:.4f}')
    print(f'Loss log  -> {loss_log}')
    print(f'Checkpoint -> {MODEL_DIR}/vqvae_best.pth')
    print()
    print('To sample after training:')
    print('  model = VQVAE.load("models/.../vqvae_best.pth")')
    print('  codes = model.active_codes(loader, device)   # only used codes')
    print('  idx   = torch.tensor(random.choices(codes, k=N))')
    print('  imgs  = model.decode_code(idx)')

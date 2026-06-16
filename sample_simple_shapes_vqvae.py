"""
Sample from a trained VQ-VAE (train_simple_shapes_vqvae.py).

Each generated image requires a 4×4 grid of code indices (16 positions).
Sampling strategy: for each position, independently pick a random active code.
This ignores spatial correlations but is the simplest valid approach.

Usage:
    cd /Users/admin/workspace/diffusion_hallu/torch-vae
    python sample_simple_shapes_vqvae.py
"""

import os
import sys
import time
import random
import math
import multiprocessing as mp

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_simple_shapes_vqvae import VQVAE, K, D, C, BETA, LATENT_H, LATENT_W, DATASET_DIR as _DATASET_DIR
from train_simple_shapes import SimpleShapesDataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH  = './models/simple-shapes-5k-16x16-vqvae/vqvae_best.pth'
DATASET_DIR = _DATASET_DIR
OUT_DIR     = './figures/simple-shapes-5k-16x16-vqvae'
N_SAMPLES   = 100_000

BATCH_SIZE  = 250
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _to_pil(tensor):
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


def _get_active_codes():
    """Encode all training images, return sorted list of active code indices."""
    model   = VQVAE.load(MODEL_PATH, K=K, D=D, C=C, beta=BETA)
    dataset = SimpleShapesDataset(DATASET_DIR)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    return model.active_codes(loader, device=torch.device('cpu'))


def _sample_worker(start_idx, count, seed, active_codes, samples_dir, counter):
    """Generate `count` images, each from a random 4×4 code grid."""
    torch.set_num_threads(1)
    random.seed(seed)

    model = VQVAE.load(MODEL_PATH, K=K, D=D, C=C, beta=BETA)

    with torch.no_grad():
        for offset in range(0, count, BATCH_SIZE):
            bs = min(BATCH_SIZE, count - offset)
            # Sample a 4×4 grid of active codes independently for each image
            idx = torch.tensor(
                [[[random.choice(active_codes) for _ in range(LATENT_W)]
                  for _ in range(LATENT_H)]
                 for _ in range(bs)],
                dtype=torch.long,
            )   # (B, 4, 4)
            imgs = model.decode_codes(idx)   # (B, 3, 16, 16)
            for i, img_t in enumerate(imgs):
                _to_pil(img_t).save(
                    os.path.join(samples_dir, f'{start_idx + offset + i:05d}.png'))
            with counter.get_lock():
                counter.value += bs


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    samples_dir = os.path.join(OUT_DIR, 'random_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # ── 1. Active codes ──────────────────────────────────────────────────────
    print('Finding active codebook entries...')
    t0 = time.time()
    active_codes = _get_active_codes()
    n_active = len(active_codes)
    print(f'  Active: {n_active}/{K} codes ({100*n_active/K:.1f}%)  in {time.time()-t0:.1f}s')

    # ── 2. Parallel sampling ──────────────────────────────────────────────────
    base_seed = time.time_ns() % (2 ** 31)
    print(f'\nSampling {N_SAMPLES:,} images via {NUM_WORKERS} workers '
          f'(batch={BATCH_SIZE}, seed={base_seed})...')

    counter = mp.Value('i', 0)
    chunk   = N_SAMPLES // NUM_WORKERS
    procs, start = [], 0
    for w in range(NUM_WORKERS):
        cnt = chunk if w < NUM_WORKERS - 1 else N_SAMPLES - start
        p = mp.Process(
            target=_sample_worker,
            args=(start, cnt, base_seed + w, active_codes, samples_dir, counter),
        )
        p.start()
        procs.append(p)
        start += cnt

    t0 = time.time()
    while any(p.is_alive() for p in procs):
        time.sleep(1)
        done    = counter.value
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (N_SAMPLES - done) / rate if rate > 0 else float('inf')
        print(f'\r  {done:,}/{N_SAMPLES:,}  ({rate:.1f} img/s, ETA {eta:.0f}s)   ',
              end='', flush=True)
    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f'\r  {N_SAMPLES:,}/{N_SAMPLES:,}  done in {elapsed:.1f}s '
          f'({N_SAMPLES/elapsed:.1f} img/s)              ')
    print(f'Samples -> {samples_dir}/')

    # ── 3. Codebook grid: decode each active code as a uniform 4×4 patch ─────
    print('\nRendering codebook grid...')
    model = VQVAE.load(MODEL_PATH, K=K, D=D, C=C, beta=BETA)

    ZOOM  = 6      # 16×16 → 96×96
    PAD   = 2
    COLS  = min(32, n_active)
    ROWS  = math.ceil(n_active / COLS)
    CELL  = 16 * ZOOM + PAD
    canvas = np.full((ROWS * CELL + PAD, COLS * CELL + PAD, 3), 40, dtype=np.uint8)

    with torch.no_grad():
        for pos, code in enumerate(active_codes):
            # Fill all 16 positions with the same code → pure "atom" visualization
            idx = torch.full((1, LATENT_H, LATENT_W), code, dtype=torch.long)
            img = model.decode_codes(idx)[0]
            cell = _to_pil(img).resize((16 * ZOOM, 16 * ZOOM), resample=Image.NEAREST)
            r, c = divmod(pos, COLS)
            y, x = r * CELL + PAD, c * CELL + PAD
            canvas[y:y + 16*ZOOM, x:x + 16*ZOOM] = np.array(cell)

    grid_path = os.path.join(OUT_DIR, 'codebook_grid.png')
    Image.fromarray(canvas).save(grid_path)
    print(f'Codebook grid ({n_active} active codes, {COLS}×{ROWS}) -> {grid_path}')

    print('\nDone.')

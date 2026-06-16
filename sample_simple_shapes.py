"""
Sample / reconstruct from a trained VAE on simple-shapes-16x16 (RGB).

Usage:
    cd /Users/admin/workspace/torch-vae
    python sample_simple_shapes.py
"""

import os
import sys
import time
import multiprocessing as mp

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/admin/workspace/diffusion_hallu/improved-diffusion')  # train_simple_shapes.py has a stale hardcoded path
from train_simple_shapes import VAE, SimpleShapesDataset

# ──────────────────────────────────────────────
# Config — must match training
# ──────────────────────────────────────────────
MODEL_PATH  = '/Users/admin/workspace/diffusion_hallu/torch-vae/models/simple-shapes-5k-16x16-col0-only-latent_dim-1/vae_simple_shapes_best.pth'
DATASET_DIR = '/Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16'
OUT_DIR     = './figures/simple-shapes-5k-16x16-col0-only-latent_dim-1'
LATENT_DIMS = 1
C           = 128
NUM_HEADS   = 4
N_SAMPLES   = 100000

# ── Parallel CPU sampling ──
BATCH_SIZE  = 250                                  # images decoded per forward pass
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)    # leave one core free for the OS


def to_pil(tensor):
    """(3, H, W) float tensor [0,1] → PIL RGB image."""
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


def _sample_worker(start_idx, count, seed, samples_dir, counter):
    """Decode `count` images starting at global index `start_idx`, save as PNGs."""
    torch.set_num_threads(1)  # avoid oversubscribing CPU cores across processes

    device = torch.device('cpu')
    model = VAE.load(MODEL_PATH, latent_dims=LATENT_DIMS, c=C, num_heads=NUM_HEADS, device=device)

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for offset in range(0, count, BATCH_SIZE):
            bs = min(BATCH_SIZE, count - offset)
            z = torch.randn(bs, LATENT_DIMS, generator=gen)
            generated = model.decode(z)   # (bs, 3, 16, 16)
            for i, img_t in enumerate(generated):
                idx = start_idx + offset + i
                to_pil(img_t).save(os.path.join(samples_dir, f'{idx:05d}.png'))
            with counter.get_lock():
                counter.value += bs


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Random samples from prior z ~ N(0, I), split across CPU worker processes ──
    samples_dir = os.path.join(OUT_DIR, 'random_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # different base seed every run -> different samples each time the script is run
    base_seed = time.time_ns() % (2 ** 31)
    print(f'Sampling {N_SAMPLES} images using {NUM_WORKERS} worker processes '
          f'(batch size {BATCH_SIZE}, base_seed={base_seed})...')

    counter = mp.Value('i', 0)
    chunk = N_SAMPLES // NUM_WORKERS
    procs, start = [], 0
    for w in range(NUM_WORKERS):
        count = chunk if w < NUM_WORKERS - 1 else N_SAMPLES - start
        p = mp.Process(target=_sample_worker, args=(start, count, base_seed + w, samples_dir, counter))
        p.start()
        procs.append(p)
        start += count

    t0 = time.time()
    while any(p.is_alive() for p in procs):
        time.sleep(1)
        done = counter.value
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (N_SAMPLES - done) / rate if rate > 0 else float('inf')
        print(f'\r  {done}/{N_SAMPLES}  ({rate:.1f} img/s, ETA {eta:.0f}s)   ', end='', flush=True)
    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f'\r  {N_SAMPLES}/{N_SAMPLES}  done in {elapsed:.1f}s '
          f'({N_SAMPLES / elapsed:.1f} img/s)               ')
    print(f'Random samples ({N_SAMPLES}) -> {samples_dir}/')

    # # ── 2. Reconstruct real images ──
    # recon_dir = os.path.join(OUT_DIR, 'reconstructions')
    # os.makedirs(recon_dir, exist_ok=True)

    # dataset = SimpleShapesDataset(DATASET_DIR)
    # n_recon = 32
    # orig_batch = torch.stack([dataset[i][0] for i in range(n_recon)]).to(device)

    # with torch.no_grad():
    #     recon_batch = model(orig_batch)   # (N, 3, 16, 16)

    # for i in range(n_recon):
    #     to_pil(orig_batch[i]).save(os.path.join(recon_dir, f'{i:04d}_orig.png'))
    #     to_pil(recon_batch[i]).save(os.path.join(recon_dir, f'{i:04d}_recon.png'))
    # print(f'Reconstructions ({n_recon} pairs) -> {recon_dir}/')

    # # ── 3. Interpolate between 2 random latent points ──
    # interp_dir = os.path.join(OUT_DIR, 'interpolations')
    # os.makedirs(interp_dir, exist_ok=True)

    # n_steps = 10
    # with torch.no_grad():
    #     z_a = torch.randn(1, LATENT_DIMS, device=device)
    #     z_b = torch.randn(1, LATENT_DIMS, device=device)
    #     frames = []
    #     for t in np.linspace(0, 1, n_steps):
    #         z_interp = (1 - t) * z_a + t * z_b
    #         img_t = model.decode(z_interp).squeeze(0)   # (3, 16, 16)
    #         frames.append(to_pil(img_t))

    # # xếp thành 1 hàng, ảnh đầu bên trái, ảnh cuối bên phải
    # W, H = frames[0].size
    # strip = Image.new('RGB', (W * n_steps, H))
    # for i, frame in enumerate(frames):
    #     strip.paste(frame, (i * W, 0))
    # strip.save(os.path.join(interp_dir, 'interpolation_strip.png'))
    # print(f'Interpolation strip -> {interp_dir}/interpolation_strip.png')

    print('\nDone.')

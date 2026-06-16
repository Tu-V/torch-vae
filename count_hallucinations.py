"""
Count hallucinations among VAE-generated simple-shapes-16x16 samples.

Reuses the shared detector from flow_matching/hallucination_detector.py
(same column-layout / double-col / empty-image logic used across the
diffusion and flow-matching pipelines).

Usage:
    cd /Users/admin/workspace/diffusion_hallu/torch-vae
    python count_hallucinations.py
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
from PIL import Image

sys.path.insert(0, '/Users/admin/workspace/diffusion_hallu/flow_matching')
from hallucination_detector import analyze_batch, summarize, COLUMN_NAMES

SAMPLES_DIR = '/Users/admin/workspace/diffusion_hallu/torch-vae/figures/simple-shapes-5k-16x16-col0-only-latent_dim-1/random_samples'
OUT_DIR     = '/Users/admin/workspace/diffusion_hallu/torch-vae/figures/simple-shapes-5k-16x16-col0-only-latent_dim-1'
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _load_and_analyze(paths):
    imgs = np.stack([np.array(Image.open(p).convert('RGB')) for p in paths])
    return analyze_batch(imgs)


if __name__ == '__main__':
    # only the new 5-digit samples (00000.png .. 99999.png) — skip stale 4-digit leftovers
    paths = sorted(
        os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR)
        if len(f) == 9 and f.endswith('.png')
    )
    n = len(paths)
    print(f'Found {n} images in {SAMPLES_DIR}')

    chunks = [c.tolist() for c in np.array_split(paths, NUM_WORKERS) if len(c) > 0]

    t0 = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        chunk_results = pool.map(_load_and_analyze, chunks)
    results = [r for chunk in chunk_results for r in chunk]
    print(f'Analyzed {len(results)} images in {time.time() - t0:.1f}s')

    s = summarize(results)

    print(f'\n=== Results ===')
    print(f"Hallucinations : {s['n_hall']} / {s['n_total']}  ({100 * s['hall_rate']:.2f}%)")
    print(f"  +- empty image (0 shapes)  : {s['n_empty']}")
    print(f"  +- double col (2+ in 1 col): {s['n_double_col']}")
    print(f"Normal         : {s['n_normal']} / {s['n_total']}")
    print()
    print(f"  {'column':10s} {'0 shapes':>10} {'1 shape':>10} {'2+ shapes':>10}")
    for name in COLUMN_NAMES:
        cc = s['col_counts'][name]
        print(f"  {name:10s} {cc['0']:>10} {cc['1']:>10} {cc['2+']:>10}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUT_DIR, 'hallucination_indices.txt'), s['hall_indices'], fmt='%d')
    np.savetxt(os.path.join(OUT_DIR, 'empty_indices.txt'), s['empty_indices'], fmt='%d')
    np.savetxt(os.path.join(OUT_DIR, 'double_col_indices.txt'), s['double_col_indices'], fmt='%d')
    print(f'\nSaved hallucination/empty/double_col indices -> {OUT_DIR}/')

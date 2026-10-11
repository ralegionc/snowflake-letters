"""Pull the matched snowflakes out of the MASCDB zarr into a compact uint8 array.

Each camera view is cropped to its bounding box, padded to a square, and resized to SIZE x SIZE.
Absolute size is lost by this step, so log(Dmax) goes in as a scalar alongside the image.
Output: data/images.npy with shape (N, 3, SIZE, SIZE) in the row order of data/table.parquet.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
SIZE = 96
ZIP = D / "mascdb/MASCdb.zarr.zip"


def open_array():
    store = zarr.storage.ZipStore(str(ZIP), mode="r")
    return zarr.open_group(store, path="MASCdb.zarr", mode="r", zarr_format=2)["data"]


def crop(img):
    ys, xs = np.nonzero(img > 0)
    if len(ys) == 0:
        return np.zeros((SIZE, SIZE), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0, 16)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = np.zeros((side, side), np.uint8)
    sy0, sx0 = cy - side // 2, cx - side // 2
    src = img[max(sy0, 0):sy0 + side, max(sx0, 0):sx0 + side]
    canvas[max(-sy0, 0):max(-sy0, 0) + src.shape[0], max(-sx0, 0):max(-sx0, 0) + src.shape[1]] = src
    return np.asarray(Image.fromarray(canvas).resize((SIZE, SIZE), Image.BILINEAR))


def work(args):
    chunk_start, rows = args  # rows: list of (out_pos, zarr_idx)
    arr = open_array()
    idx = np.array([r[1] for r in rows])
    block = arr[idx.min():idx.max() + 1]
    out = np.zeros((len(rows), 3, SIZE, SIZE), np.uint8)
    for k, (_, zi) in enumerate(rows):
        trip = block[zi - idx.min()]
        for c in range(3):
            out[k, c] = crop(trip[..., c])
    return [r[0] for r in rows], out


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t = pd.read_parquet(D / "table.parquet")
    arr = open_array()
    print("zarr", arr.shape, arr.chunks, arr.dtype, flush=True)
    chunk = arr.chunks[0]
    zi = t.zarr_idx.to_numpy()
    jobs = {}
    for pos, z in enumerate(zi):
        jobs.setdefault(z // chunk, []).append((pos, z))
    out = np.lib.format.open_memmap(D / "images.npy", mode="w+", dtype=np.uint8, shape=(len(t), 3, SIZE, SIZE))
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for poss, imgs in ex.map(work, sorted(jobs.items()), chunksize=1):
            out[poss] = imgs
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} chunks", flush=True)
    out.flush()
    print("done", out.shape)

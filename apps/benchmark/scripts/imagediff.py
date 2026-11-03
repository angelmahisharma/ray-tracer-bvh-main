#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np

def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )

def read_pfm(path):
    path = Path(path)
    if path.suffix.lower() == ".png":
        raise ValueError(f"{path} is a PNG, not a PFM. PFM is required.")
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        channels = 3 if header == "PF" else 1 if header == "Pf" else 0
        if channels == 0:
            raise ValueError(f"{path}: not a PFM (header={header!r})")
        dims = f.readline().decode("ascii").strip().split()
        width, height = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)
        count = width * height * channels
        raw = f.read(count * 4)
        data = np.frombuffer(raw, dtype=endian + "f4")
    img = data.reshape(height, width, channels) if channels == 3 else data.reshape(height, width)
    return np.array(np.flipud(img) * scale, dtype=np.float32)

def compute_rmse(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    n_bad = int(np.sum(~finite))
    if not np.any(finite): return float("nan"), float("nan"), n_bad
    diff = a[finite].astype(np.float64) - b[finite].astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_err = float(np.max(np.abs(diff)))
    return rmse, max_err, n_bad

def compute_ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return None
    combined = np.concatenate([a.ravel(), b.ravel()])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0: return None
    data_range = max(float(np.percentile(combined, 99.5)), 1e-6)
    a_c = np.clip(a, 0.0, data_range)
    b_c = np.clip(b, 0.0, data_range)
    if a_c.ndim == 2:
        return float(ssim_fn(a_c, b_c, data_range=data_range))
    return float(ssim_fn(a_c, b_c, data_range=data_range, channel_axis=-1))

def save_diff_png(a, b, path, scale=10.0, linear=False):
    try:
        from PIL import Image
    except ImportError:
        return
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    diff = np.where(np.isfinite(diff), diff, 1.0 / max(scale, 1e-9)) * scale
    diff = srgb_encode(diff) if not linear else np.clip(diff, 0.0, 1.0)
    img8 = np.round(diff * 255.0).astype(np.uint8)
    if img8.ndim == 2: img8 = np.stack([img8] * 3, axis=-1)
    Image.fromarray(img8).save(path)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--diff", metavar="OUT.png")
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--linear", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        a = read_pfm(args.a); b = read_pfm(args.b)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1
    if a.shape != b.shape:
        print(f"ERROR: shape mismatch", file=sys.stderr); return 2

    rmse, max_err, n_bad = compute_rmse(a, b)
    ssim_val = compute_ssim(a, b)

    if args.diff: save_diff_png(a, b, args.diff, scale=args.scale, linear=args.linear)
    if args.json:
        print(json.dumps({"rmse": rmse, "ssim": ssim_val, "max_abs_err": max_err, "n_nonfinite": n_bad}))
    else:
        print(f"RMSE:   {rmse:.6f}\nSSIM:   {ssim_val if ssim_val else 'N/A'}")
    if args.threshold is not None and rmse >= args.threshold: return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
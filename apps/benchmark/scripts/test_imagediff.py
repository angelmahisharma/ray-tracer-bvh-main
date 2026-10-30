#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGEDIFF = HERE / "imagediff.py"

def write_pfm(path, img):
    # Writes a little-endian PFM (negative scale indicates little-endian)
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    if img.ndim == 2:
        header, channels = "Pf", 1
    else:
        header, channels = "PF", img.shape[2]
    flipped = np.flipud(img) # PFM is bottom-up
    with open(path, "wb") as f:
        f.write(f"{header}\n".encode())
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")
        f.write(flipped.tobytes(order="C"))

def run_imagediff(a, b):
    return subprocess.run(
        [sys.executable, str(IMAGEDIFF), str(a), str(b), "--json"],
        capture_output=True, text=True,
    )

def test_identical(tmpdir):
    # Same image vs itself should be a perfect match
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 2, size=(32, 48, 3)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, img)
    write_pfm(b, img)

    r = run_imagediff(a, b)
    assert r.returncode == 0, r.stderr
    m = json.loads(r.stdout)
    print(f"  RMSE={m['rmse']:.6f}  SSIM={m['ssim']}")
    assert m["rmse"] == 0.0
    if m["ssim"] is not None:
        assert abs(m["ssim"] - 1.0) < 1e-9
    print("  PASS")

def test_noise_floor(tmpdir):
    # Two images with Gaussian noise should yield an RMSE roughly equal to the noise sigma
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 2, size=(32, 48, 3)).astype(np.float32)
    sigma = 0.02
    noisy = (base + rng.normal(0, sigma, base.shape)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, base)
    write_pfm(b, noisy)

    r = run_imagediff(a, b)
    m = json.loads(r.stdout)
    print(f"  RMSE={m['rmse']:.6f}  (expected ≈ {sigma})")
    assert 0.015 < m["rmse"] < 0.025
    print("  PASS")

def test_flip(tmpdir):
    # Structural differences (like a vertical flip) should yield high RMSE and low SSIM
    rng = np.random.default_rng(2)
    img = rng.uniform(0, 2, size=(64, 48, 3)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, img)
    write_pfm(b, np.flipud(img))

    r = run_imagediff(a, b)
    m = json.loads(r.stdout)
    print(f"  RMSE={m['rmse']:.6f}  SSIM={m['ssim']}")
    assert m["rmse"] > 0.5
    if m["ssim"] is not None:
        assert m["ssim"] < 0.3
    print("  PASS")

def test_shape_mismatch(tmpdir):
    # Comparing a 10x10 image against a 20x20 image should fail gracefully with exit code 2
    rng = np.random.default_rng(3)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, rng.uniform(0, 1, size=(10, 10, 3)).astype(np.float32))
    write_pfm(b, rng.uniform(0, 1, size=(20, 20, 3)).astype(np.float32))
    
    r = run_imagediff(a, b)
    assert r.returncode == 2
    print(f"  ERROR message: {r.stderr.strip()}")
    print("  PASS")

def main():
    tests = [
        ("identical images → RMSE 0, SSIM 1", test_identical),
        ("noise floor ≈ sigma",               test_noise_floor),
        ("flipped image → high RMSE, low SSIM", test_flip),
        ("shape mismatch → exit 2",           test_shape_mismatch),
    ]
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        for name, fn in tests:
            print(f"\n[TEST] {name}")
            try:
                fn(tmpdir)
            except AssertionError as e:
                print(f"  FAIL: {e}")
                failed += 1
    print(f"\n{'='*60}")
    print(f"{len(tests)-failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
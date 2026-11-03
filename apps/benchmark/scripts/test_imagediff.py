#!/usr/bin/env python3
import subprocess, sys, tempfile, json, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGEDIFF = HERE / "imagediff.py"

def write_pfm(path, img):
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    header = "Pf" if img.ndim == 2 else "PF"
    with open(path, "wb") as f:
        f.write(f"{header}\n{w} {h}\n-1.0\n".encode())
        f.write(np.flipud(img).tobytes(order="C"))

def run_imagediff(a, b):
    return subprocess.run([sys.executable, str(IMAGEDIFF), str(a), str(b), "--json"], capture_output=True, text=True)

def test_identical(tmpdir):
    img = np.random.default_rng(0).uniform(0, 2, size=(32, 48, 3)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, img); write_pfm(b, img)
    m = json.loads(run_imagediff(a, b).stdout)
    assert m["rmse"] == 0.0
    if m["ssim"] is not None: assert abs(m["ssim"] - 1.0) < 1e-9
    print("  PASS")

def test_noise_floor(tmpdir):
    base = np.random.default_rng(1).uniform(0, 2, size=(32, 48, 3)).astype(np.float32)
    noisy = (base + np.random.default_rng(1).normal(0, 0.02, base.shape)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, base); write_pfm(b, noisy)
    m = json.loads(run_imagediff(a, b).stdout)
    assert 0.015 < m["rmse"] < 0.025
    print("  PASS")

def test_flip(tmpdir):
    img = np.random.default_rng(2).uniform(0, 2, size=(64, 48, 3)).astype(np.float32)
    a, b = tmpdir / "a.pfm", tmpdir / "b.pfm"
    write_pfm(a, img); write_pfm(b, np.flipud(img))
    m = json.loads(run_imagediff(a, b).stdout)
    assert m["rmse"] > 0.5
    if m["ssim"] is not None: assert m["ssim"] < 0.3
    print("  PASS")

def main():
    tests = [("identical", test_identical), ("noise", test_noise_floor), ("flip", test_flip)]
    with tempfile.TemporaryDirectory() as d:
        for name, fn in tests:
            print(f"[TEST] {name}")
            fn(Path(d))
    print("All tests passed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path

# Correct paths for the Windows Visual Studio build
RENDERER = "build/apps/cpu_renderer/Release/cpu_renderer.exe"
SCENE = "scenes/part7_cornell/cornell.scene"
REF_SPP = 64  # Lowered for quick testing, can be raised later
SPP_LIST = [4, 8, 16, 32]
OUT_DIR = Path("noise_check_out")

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}:")
        print(result.stderr)
        sys.exit(1)

def main():
    if not Path(RENDERER).exists():
        print(f"Renderer not found at {RENDERER}. Check the path!")
        sys.exit(1)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"\n--- Rendering Reference ({REF_SPP} spp) ---")
    ref_pfm = OUT_DIR / f"render_{REF_SPP}.pfm"
    run_cmd([RENDERER, SCENE, str(OUT_DIR / f"render_{REF_SPP}"), str(REF_SPP), "5"])

    print("\n--- Sweeping spp and computing RMSE ---")
    results = []
    imagediff_script = Path("apps/benchmark/scripts/imagediff.py")

    for spp in SPP_LIST:
        pfm_path = OUT_DIR / f"render_{spp}.pfm"
        run_cmd([RENDERER, SCENE, str(OUT_DIR / f"render_{spp}"), str(spp), "5"])

        diff_cmd = [sys.executable, str(imagediff_script), str(pfm_path), str(ref_pfm), "--json"]
        res = subprocess.run(diff_cmd, capture_output=True, text=True)

        if res.returncode != 0:
            print(f"imagediff failed: {res.stderr}")
            sys.exit(1)

        data = json.loads(res.stdout)
        rmse = data["rmse"]
        results.append((spp, rmse))
        print(f" {spp:4d} spp -> RMSE: {rmse:.4f}")

    print("\n### Noise Check Results\n")
    print("| spp | RMSE vs Reference |")
    print("|-----|-------------------|")
    for spp, rmse in results:
        print(f"| {spp:3d} | {rmse:.4f} |")
    print("\n*Rule of thumb: pick the lowest spp where RMSE < 0.02.*")

if __name__ == "__main__":
    main()
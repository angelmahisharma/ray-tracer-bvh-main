#!/usr/bin/env python3
import subprocess
import csv
import sys
import re
from pathlib import Path

OUTPUT_CSV = "benchmark_results.csv"

# The renderers you built
RENDERERS = [
    ("build/apps/cpu_renderer/Release/cpu_renderer.exe", "CPU Brute"),
    ("build/apps/cuda_renderer/Release/cuda_renderer.exe", "CUDA Brute"),
    ("build/apps/cuda_bvh_renderer/Release/cuda_bvh_renderer.exe", "CUDA BVH"),
    ("build/apps/optix_renderer/Release/optix_renderer.exe", "OptiX"),
]

# The SPP sweep
SPP_LIST = [1, 2, 4, 8, 16, 32]

def run_once(renderer, scene, spp, bounces, run_index, name):
    safe_name = name.replace(" ", "_")
    out_prefix = f"bench_{safe_name}_spp{spp}_{run_index}"
    
    result = subprocess.run(
        [renderer, scene, out_prefix, str(spp), str(bounces)],
        capture_output=True, text=True
    )
    
    match = re.search(r"Render time:\s+([\d.]+)\s+s", result.stdout)
    if not match:
        print(f"\nError: Could not find 'Render time' in {name} output!")
        return None
    return float(match.group(1)) * 1000.0

def main():
    print(f"--- Starting SPP Sweep Benchmark ---")
    
    file_exists = Path(OUTPUT_CSV).exists()
    csv_file = open(OUTPUT_CSV, mode="a", newline="")
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(["renderer_name", "scene", "spp", "bounces", "median_ms", "min_ms"])

    for spp in SPP_LIST:
        for renderer, name in RENDERERS:
            # Skip CPU at 32 spp
            if name == "CPU Brute" and spp == 32:
                print(f"\nSkipping CPU Brute at {spp} SPP to save time.")
                continue

            if not Path(renderer).exists():
                continue

            print(f"\n--- Benchmarking {name} at {spp} SPP ---")
            
            # Warm-up
            print(f"  Warm-up...", end="", flush=True)
            run_once(renderer, "scenes/part7_cornell/cornell.scene", spp, 5, "warmup", name)
            print(" Done.")

            times = []
            for i in range(5):
                print(f"  Run {i+1}/5...", end="", flush=True)
                ms = run_once(renderer, "scenes/part7_cornell/cornell.scene", spp, 5, i, name)
                if ms is not None:
                    print(f" {ms:.2f} ms")
                    times.append(ms)

            if not times: continue
            times.sort()
            median = times[len(times) // 2]
            min_time = min(times)
            
            print(f"  Result: Median = {median:.2f} ms")
            writer.writerow([name, "scenes/part7_cornell/cornell.scene", spp, 5, f"{median:.2f}", f"{min_time:.2f}"])
            csv_file.flush()

    csv_file.close()
    print(f"\n✅ Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
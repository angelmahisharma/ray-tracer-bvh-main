#!/usr/bin/env python3
import subprocess
import csv
import sys
import re
from pathlib import Path

OUTPUT_CSV = "benchmark_results.csv"

# Format: (Renderer Path, Scene Path, SPP, Bounces, Friendly Name, Extra Args)
# Note: CPU Brute, CUDA, and OptiX are commented out because we already have their data.
# We are only running the new CPU BVH.
JOBS = [
    # ("build/apps/cpu_renderer/Release/cpu_renderer.exe", "scenes/part7_cornell/cornell.scene", 16, 5, "CPU Brute", []),
    ("build/apps/cpu_renderer/Release/cpu_renderer.exe", "scenes/part7_cornell/cornell.scene", 16, 5, "CPU BVH", ["sah"]),
    # ("build/apps/cuda_renderer/Release/cuda_renderer.exe", "scenes/part7_cornell/cornell.scene", 16, 5, "CUDA Brute", []),
    # ("build/apps/cuda_bvh_renderer/Release/cuda_bvh_renderer.exe", "scenes/part7_cornell/cornell.scene", 16, 5, "CUDA BVH", []),
    # ("build/apps/optix_renderer/Release/optix_renderer.exe", "scenes/part7_cornell/cornell.scene", 16, 5, "OptiX", []),
]

def run_once(renderer, scene, spp, bounces, run_index, name, extra_args):
    """Runs the renderer once and extracts the kernel time from stdout."""
    safe_name = name.replace(" ", "_")
    out_prefix = f"bench_{safe_name}_spp{spp}_{run_index}"
    
    # Combine the base command with any extra arguments (like 'sah')
    cmd = [renderer, scene, out_prefix, str(spp), str(bounces)] + extra_args
    
    # Run the executable and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Look for: "Render time: 12.345 s" in the C++ output
    match = re.search(r"Render time:\s+([\d.]+)\s+s", result.stdout)
    if not match:
        print(f"\nError: Could not find 'Render time' in {name} output!")
        print("Output was:", result.stdout)
        return None
        
    # Convert seconds to milliseconds
    return float(match.group(1)) * 1000.0

def main():
    print(f"--- Starting Benchmark Suite ---")
    
    file_exists = Path(OUTPUT_CSV).exists()
    csv_file = open(OUTPUT_CSV, mode="a", newline="")
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(["renderer_name", "scene", "spp", "bounces", "median_ms", "min_ms"])

    for renderer, scene, spp, bounces, name, extra_args in JOBS:
        if not Path(renderer).exists():
            print(f"Skipping {name}: Renderer not found at {renderer}")
            continue

        print(f"\n--- Benchmarking {name} ---")
        
        # 1. Warm-up
        print(f"  Performing warm-up run...", end="", flush=True)
        run_once(renderer, scene, spp, bounces, "warmup", name, extra_args)
        print(" Done.")

        # 2. Timed runs
        times = []
        for i in range(5):
            print(f"  Run {i+1}/5...", end="", flush=True)
            ms = run_once(renderer, scene, spp, bounces, i, name, extra_args)
            if ms is not None:
                print(f" {ms:.2f} ms")
                times.append(ms)

        if not times:
            continue

        # 3. Calculate stats
        times.sort()
        median = times[len(times) // 2]
        min_time = min(times)
        
        print(f"  Result: Median = {median:.2f} ms | Min = {min_time:.2f} ms")

        # 4. Save to CSV
        writer.writerow([name, scene, spp, bounces, f"{median:.2f}", f"{min_time:.2f}"])
        csv_file.flush()

    csv_file.close()
    print(f"\n✅ Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
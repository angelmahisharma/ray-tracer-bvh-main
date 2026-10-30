#!/usr/bin/env python3
import subprocess
import csv
import sys
import re
from pathlib import Path

# --- Configuration ---
OUTPUT_CSV = "benchmark_results.csv"

# JOBS format: (Renderer Path, Scene Path, SPP, Bounces, Friendly Name)
# We use SPP=4 for now so the CPU doesn't take 10 minutes per run!
JOBS = [
    ("build/apps/cpu_renderer/cpu_renderer.exe", "scenes/part7_cornell/cornell.scene", 4, 5, "CPU Brute (Cornell)"),
]

def run_once(renderer, scene, spp, bounces, run_index):
    """Runs the renderer and extracts ONLY the kernel time from stdout."""
    out_prefix = f"bench_out_{run_index}"
    
    result = subprocess.run(
        [renderer, scene, out_prefix, str(spp), str(bounces)],
        capture_output=True, text=True
    )
    
    # Extract the exact line: "Render time: 12.345 s"
    match = re.search(r"Render time:\s+([\d.]+)\s+s", result.stdout)
    if not match:
        print(f"\nError: Could not find 'Render time' in output!")
        print(result.stdout)
        return None
        
    elapsed_seconds = float(match.group(1))
    return elapsed_seconds * 1000.0  # Convert to milliseconds

def main():
    print(f"--- Starting Benchmark Suite ---")
    
    file_exists = Path(OUTPUT_CSV).exists()
    csv_file = open(OUTPUT_CSV, mode="a", newline="")
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(["renderer_name", "scene", "spp", "bounces", "median_ms", "min_ms"])

    for renderer, scene, spp, bounces, name in JOBS:
        if not Path(renderer).exists():
            print(f"Skipping {name}: Renderer not found at {renderer}")
            continue

        print(f"\n--- Benchmarking {name} ---")
        print(f"Scene: {scene} | SPP: {spp}")

        # 1. Warm-up run (PDF: "Do one warm-up run")
        print("  Performing warm-up run...", end="", flush=True)
        run_once(renderer, scene, spp, bounces, "warmup")
        print(" Done.")

        # 2. Timed runs (PDF: "average at least 5 runs")
        times = []
        for i in range(5):
            print(f"  Run {i+1}/5...", end="", flush=True)
            ms = run_once(renderer, scene, spp, bounces, i)
            if ms is not None:
                print(f" {ms:.2f} ms")
                times.append(ms)
            else:
                print(" Failed.")

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
    print(f"\n✅ Done! Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
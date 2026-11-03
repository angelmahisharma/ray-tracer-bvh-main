#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Prevents Windows from hanging
import matplotlib.pyplot as plt
import csv
from pathlib import Path

CSV_FILE = "benchmark_results.csv"
OUT_DIR = Path("report_charts")

def load_data():
    """Reads the benchmark CSV and returns a dictionary of renderer data."""
    data = {}
    if not Path(CSV_FILE).exists():
        print(f"Error: {CSV_FILE} not found! Run the benchmark first.")
        return data

    with open(CSV_FILE, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["renderer_name"]
            spp = int(row["spp"])
            median_ms = float(row["median_ms"])

            if name not in data:
                data[name] = {"spp": [], "time": []}

            data[name]["spp"].append(spp)
            data[name]["time"].append(median_ms)

    # Sort data by SPP so the lines connect properly
    for name in data:
        sorted_indices = sorted(range(len(data[name]["spp"])), key=lambda i: data[name]["spp"][i])
        data[name]["spp"] = [data[name]["spp"][i] for i in sorted_indices]
        data[name]["time"] = [data[name]["time"][i] for i in sorted_indices]

    return data

def make_spp_chart(data):
    if not data: return
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'CPU Brute': 'red', 'CUDA Brute': 'orange', 'CUDA BVH': 'blue', 'OptiX': 'green'}
    markers = {'CPU Brute': 'o', 'CUDA Brute': 's', 'CUDA BVH': '^', 'OptiX': 'd'}

    for name, stats in data.items():
        ax.plot(stats["spp"], stats["time"], marker=markers.get(name, 'x'),
                label=name, color=colors.get(name, 'black'), linewidth=2, markersize=8)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    ax.set_xlabel('Samples Per Pixel (SPP) - Log Scale', fontsize=12)
    ax.set_ylabel('Render Time (ms) - Log Scale', fontsize=12)
    ax.set_title('Render Time vs. Sample Size (Cornell Box)', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(title='Renderer', fontsize=11)

    out_path = OUT_DIR / 'chart3_time_vs_spp.png'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved chart to {out_path}")
    plt.close(fig)

def make_speedup_chart(data):
    if "CPU Brute" not in data: return
    OUT_DIR.mkdir(exist_ok=True)

    # Find the time at 16 SPP for the CPU baseline
    baseline = next((t for s, t in zip(data["CPU Brute"]["spp"], data["CPU Brute"]["time"]) if s == 16), None)
    if not baseline:
        print("Could not find 16 SPP data for CPU Brute.")
        return

    names = []
    speedups = []

    for name, stats in data.items():
        if name == "CPU Brute": continue
        # Find time at 16 SPP
        time_at_16 = next((t for s, t in zip(stats["spp"], stats["time"]) if s == 16), None)
        if time_at_16:
            names.append(name)
            speedups.append(baseline / time_at_16)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, speedups, color=['orange', 'blue', 'green'])
    ax.set_yscale('log')
    ax.set_ylabel('Speedup vs CPU Brute Force (Log Scale)', fontsize=12)
    ax.set_title('Speedup at 16 SPP (Cornell Box)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.1, f'{yval:,.0f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    out_path = OUT_DIR / 'chart2_speedup.png'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved chart to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    if data:
        print("Generating charts...")
        make_spp_chart(data)
        make_speedup_chart(data)
        print("Done! Open the 'report_charts' folder.")
    else:
        print("No data to plot.")
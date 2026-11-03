#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg') # Prevents Windows from hanging
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("report_charts")

# Hardcoded data from your exact benchmark run
DATA = {
    "CPU Brute": {
        "spp": [1, 2, 4, 8, 16],
        "time": [9855.57, 17566.20, 33599.60, 66773.30, 133545.00]
    },
    "CUDA Brute": {
        "spp": [1, 2, 4, 8, 16, 32],
        "time": [163.75, 318.94, 642.31, 1352.66, 2811.14, 5957.41]
    },
    "CUDA BVH": {
        "spp": [1, 2, 4, 8, 16, 32],
        "time": [5.34, 9.61, 18.78, 37.01, 75.17, 166.37]
    },
    "OptiX": {
        "spp": [1, 2, 4, 8, 16, 32],
        "time": [2.26, 4.74, 9.89, 17.11, 33.66, 68.59]
    }
}

def make_spp_chart():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'CPU Brute': 'red', 'CUDA Brute': 'orange', 'CUDA BVH': 'blue', 'OptiX': 'green'}
    markers = {'CPU Brute': 'o', 'CUDA Brute': 's', 'CUDA BVH': '^', 'OptiX': 'd'}

    for name, stats in DATA.items():
        ax.plot(stats["spp"], stats["time"], marker=markers[name], 
                label=name, color=colors[name], linewidth=2, markersize=8)

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

def make_speedup_chart():
    OUT_DIR.mkdir(exist_ok=True)
    baseline_16spp = 133545.00 # CPU at 16 SPP
    
    names = ["CUDA Brute", "CUDA BVH", "OptiX"]
    speedups = [
        baseline_16spp / 2811.14,
        baseline_16spp / 75.17,
        baseline_16spp / 33.66
    ]

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
    print("Generating charts....")
    make_spp_chart()
    make_speedup_chart()
    print("Done! Open the 'report_charts' folder.")
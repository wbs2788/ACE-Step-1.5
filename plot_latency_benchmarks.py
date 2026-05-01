#!/usr/bin/env python3
"""
Visualize Latency Benchmarks for ISMIR Paper
Produces publication-quality plots from the generated CSV files.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    files = sorted(glob.glob("output/ismir_benchmarks/*.csv"))
    results_p = []
    results_a = []

    for f in files:
        df = pd.read_csv(f)
        if df.empty: continue
        name = os.path.basename(f).replace(".csv", "")
        mode = df["mode"].iloc[0]
        
        if mode == "standard":
            ttfa = df["wall_time_sec"].mean()
            rtf = df["rtf_wall"].mean()
            results_p.append({
                "Experiment": name, 
                "Model": name.split("_")[0] + "_" + name.split("_")[-1],
                "TTFA (s)": ttfa, 
                "RTF": rtf
            })
        else:
            ttfa = df["conditioning_time_sec"].mean() + df["first_chunk_latency_sec"].mean()
            rtf = df["rtf_total"].mean()
            chunk_lat = df["avg_prediction_chunk_latency_sec"].mean()
            
            if "1.0s" in name or "ours" in name: chunk_dur = 1.0
            elif "0.5s" in name: chunk_dur = 0.5
            elif "1.5s" in name: chunk_dur = 1.5
            elif "2.0s" in name: chunk_dur = 2.0
            else: chunk_dur = 1.0
            
            control_lat = chunk_dur + chunk_lat
            
            row = {
                "Experiment": name, 
                "Model": name,
                "Chunk Size (s)": chunk_dur,
                "TTFA (s)": ttfa, 
                "RTF": rtf, 
                "Control Latency (s)": control_lat
            }
            if "ablation" in name:
                results_a.append(row)
            else:
                row["Model"] = "Ours (Streaming 1s)"
                results_p.append(row)
                
    return pd.DataFrame(results_p), pd.DataFrame(results_a)


def plot_paradigm_comparison(df_p, out_dir):
    """Plot TTFA and RTF for the 4 baselines."""
    # Custom names for the paper
    name_map = {
        "baseline1_no_lora_8steps": "DiT (8 Steps)",
        "baseline2_with_lora_8steps": "DiT + LoRA (8 Steps)",
        "baseline3_with_lora_1step": "Distilled (1 Step Batch)",
        "Ours (Streaming 1s)": "Ours (1 Step Streaming)"
    }
    df_p["Model"] = df_p["Experiment"].map(lambda x: name_map.get(x, x))
    if "Ours_WithLoRA_Streaming" in df_p["Experiment"].values:
        df_p.loc[df_p["Experiment"] == "Ours_WithLoRA_Streaming", "Model"] = "Ours (1 Step Streaming)"

    # Set up the matplotlib figure
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot TTFA
    sns.barplot(x="Model", y="TTFA (s)", data=df_p, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Time To First Audio (TTFA)")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value annotations
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}s", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12, xytext=(0, 5),
                        textcoords='offset points')

    # Plot RTF
    sns.barplot(x="Model", y="RTF", data=df_p, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Real-Time Factor (RTF)")
    axes[1].set_ylabel("RTF (Lower is better)")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add a red dashed line for RTF = 1.0 (Real-time threshold)
    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=2, label="Real-time Threshold (1.0)")
    axes[1].legend()

    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12, xytext=(0, 5),
                        textcoords='offset points')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "paradigm_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def plot_chunk_ablation(df_a, out_dir):
    """Plot Trade-off between Control Latency and RTF across chunk sizes."""
    df_a = df_a.sort_values(by="Chunk Size (s)")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Chunk Size (seconds)')
    ax1.set_ylabel('Control Latency (seconds)', color=color1)
    # Plot Control Latency
    sns.lineplot(x="Chunk Size (s)", y="Control Latency (s)", data=df_a, ax=ax1, color=color1, marker='o', linewidth=3, markersize=10, label="Control Latency")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(df_a["Control Latency (s)"]) * 1.2)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Real-Time Factor (RTF)', color=color2)
    # Plot RTF
    sns.lineplot(x="Chunk Size (s)", y="RTF", data=df_a, ax=ax2, color=color2, marker='s', linewidth=3, markersize=10, label="RTF")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(df_a["RTF"]) * 1.5)

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")
    
    plt.title("Trade-off: Chunk Size vs Control Latency & RTF")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "chunk_size_ablation.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def main():
    out_dir = "output/ismir_benchmarks/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    df_p, df_a = load_data()
    
    if not df_p.empty:
        plot_paradigm_comparison(df_p, out_dir)
        
    if not df_a.empty:
        plot_chunk_ablation(df_a, out_dir)

if __name__ == "__main__":
    main()

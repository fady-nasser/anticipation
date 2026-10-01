import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, title, xlabel="Follower (Reactor)", ylabel="Leader (Trigger)", 
                 cmap="Blues", cbar_label="Anticipation Event Count", 
                 figsize=(11, 9.5), out_path=None, vmin=0):
    """Plot a single clean heatmap with explicit integer values and no blank masked squares."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, ax=ax, cmap=cmap, annot=True, fmt='d', linewidths=0.4,
                mask=None, cbar_kws={'label': cbar_label}, vmin=vmin)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig, ax

def plot_four_panel_grid(panels_dict, title, out_path=None, figsize=(22, 20)):
    """
    Plot a 2x2 grid of heatmaps.
    panels_dict: dict of {'(A) Title': df, ...} (length 4)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, (p_title, df) in enumerate(panels_dict.items()):
        r, c = positions[idx]
        ax = axes[r, c]
        sns.heatmap(df, ax=ax, cmap='Blues', annot=True, fmt='d', linewidths=0.4,
                    mask=None, cbar_kws={'label': 'Anticipation Event Count'}, vmin=0)
        ax.set_title(p_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Follower (Reactor)', fontsize=10)
        ax.set_ylabel('Leader (Trigger)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    return fig, axes

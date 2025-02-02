"""Stratified analysis: how restoration quality varies with shadow intensity and scene conditions.

Per-image attributes (computed from the ground truth, so they are independent of the model):
    shadow_intensity  1 - mean luminance(shadow img, in mask) / mean luminance(shadow-free, in mask)
                      0 = no darkening, 1 = black shadow
    shadow_area       fraction of pixels covered by the shadow mask
    scene_brightness  mean luminance of the shadow-free image
    scene_texture     mean gradient magnitude of the shadow-free luminance (smooth vs textured)

Each attribute is split into three equal-count bins (tertiles) unless fixed edges are configured.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

ATTRIBUTES = {
    "shadow_intensity": ["light", "medium", "strong"],
    "shadow_area": ["small", "medium", "large"],
    "scene_brightness": ["dark", "medium", "bright"],
    "scene_texture": ["smooth", "medium", "textured"],
}

# Colours (validated categorical palette, slots 1-2): model = blue, input baseline = orange
COLOR_MODEL = "#2a78d6"
COLOR_INPUT = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _luminance(img):
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


@torch.no_grad()
def image_attributes(shadow, gt, mask):
    """shadow/gt: (3,H,W), mask: (1,H,W) tensors in [0,1]."""
    m = mask[0] > 0.5
    ls, lg = _luminance(shadow.float()), _luminance(gt.float())
    if m.any():
        intensity = float((1.0 - ls[m].mean() / lg[m].mean().clamp_min(1e-6)).clamp(0, 1))
    else:
        intensity = float("nan")
    dx = lg[:, 1:] - lg[:, :-1]
    dy = lg[1:, :] - lg[:-1, :]
    texture = float(torch.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2).mean())
    return {
        "shadow_intensity": intensity,
        "shadow_area": float(m.float().mean()),
        "scene_brightness": float(lg.mean()),
        "scene_texture": texture,
    }


def assign_bins(df, bin_edges=None):
    """Adds '<attr>_bin' columns. bin_edges: optional {attr: [e0, e1, e2, e3]}."""
    bin_edges = bin_edges or {}
    edges_used = {}
    for attr, labels in ATTRIBUTES.items():
        col = df[attr]
        if attr in bin_edges and bin_edges[attr]:
            edges = list(bin_edges[attr])
            df[f"{attr}_bin"] = pd.cut(col, bins=edges, labels=labels, include_lowest=True)
        else:
            valid = col.dropna()
            if len(valid) < 3:
                df[f"{attr}_bin"] = np.nan
                continue
            q = np.quantile(valid, [0, 1 / 3, 2 / 3, 1]).tolist()
            edges = q
            # rank-based tertiles: robust to ties
            df[f"{attr}_bin"] = pd.qcut(col.rank(method="first"), 3, labels=labels)
        edges_used[attr] = [float(e) for e in edges]
    return df, edges_used


def stratified_summary(df, metric_cols):
    """Long table: attribute, bin, n, mean of every metric (model and input baseline)."""
    rows = []
    for attr, labels in ATTRIBUTES.items():
        col = f"{attr}_bin"
        if col not in df or df[col].isna().all():
            continue
        for lab in labels:
            sub = df[df[col] == lab]
            if len(sub) == 0:
                continue
            row = {"attribute": attr, "bin": lab, "n": len(sub),
                   f"{attr}_mean": float(sub[attr].mean())}
            for m in metric_cols:
                if m in sub:
                    row[m] = float(sub[m].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_stratified(summary, out_dir, metrics=(("psnr_all", "PSNR (dB) ↑"), ("ssim_all", "SSIM ↑"),
                                               ("mae_lab_all", "MAE LAB ↓"))):
    """One figure per attribute: grouped bars (model vs. input) for PSNR / SSIM / MAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for attr in summary["attribute"].unique():
        sub = summary[summary["attribute"] == attr]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
        x = np.arange(len(sub))
        bw = 0.38
        for ax, (m, title) in zip(np.atleast_1d(axes), metrics):
            if m not in sub or f"input_{m}" not in sub:
                continue
            b1 = ax.bar(x - bw / 2 - 0.01, sub[f"input_{m}"], bw, color=COLOR_INPUT, label="Input (shadow)")
            b2 = ax.bar(x + bw / 2 + 0.01, sub[m], bw, color=COLOR_MODEL, label="Model output")
            for bars in (b2,):  # selective direct labels: model values only
                for r in bars:
                    ax.annotate(f"{r.get_height():.3g}", (r.get_x() + r.get_width() / 2, r.get_height()),
                                ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY,
                                xytext=(0, 2), textcoords="offset points")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(sub["bin"], sub["n"])])
            ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
            _style_axes(ax)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2)
        fig.suptitle(f"Restoration quality by {attr.replace('_', ' ')}", x=0.01, ha="left",
                     fontsize=11, color=TEXT_PRIMARY)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        p = os.path.join(out_dir, f"by_{attr}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_distributions(df, out_dir):
    """PSNR histogram (input vs model) and PSNR gain vs shadow intensity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lo = np.nanmin([df["input_psnr_all"].min(), df["psnr_all"].min()])
    hi = np.nanmax([df["input_psnr_all"].max(), df["psnr_all"].max()])
    bins = np.linspace(lo, hi, 30)
    axes[0].hist(df["input_psnr_all"], bins=bins, color=COLOR_INPUT, alpha=0.75, label="Input (shadow)",
                 edgecolor="white", linewidth=0.5)
    axes[0].hist(df["psnr_all"], bins=bins, color=COLOR_MODEL, alpha=0.75, label="Model output",
                 edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("PSNR (dB)", color=TEXT_SECONDARY)
    axes[0].set_ylabel("Images", color=TEXT_SECONDARY)
    axes[0].set_title("Per-image PSNR distribution", loc="left", fontsize=10, color=TEXT_PRIMARY)
    axes[0].legend(frameon=False, fontsize=9)

    gain = df["psnr_all"] - df["input_psnr_all"]
    axes[1].scatter(df["shadow_intensity"], gain, s=14, color=COLOR_MODEL, alpha=0.7,
                    edgecolors="white", linewidths=0.5)
    axes[1].axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    axes[1].set_xlabel("Shadow intensity (0 = faint, 1 = black)", color=TEXT_SECONDARY)
    axes[1].set_ylabel("PSNR gain over input (dB)", color=TEXT_SECONDARY)
    axes[1].set_title("PSNR gain vs. shadow intensity", loc="left", fontsize=10, color=TEXT_PRIMARY)
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    p = os.path.join(out_dir, "psnr_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def run_analysis(df, out_dir, bin_edges=None):
    """Bins images, writes per-attribute CSV/JSON summaries and plots. Returns the summary table."""
    df, edges = assign_bins(df, bin_edges)
    metric_cols = [c for c in df.columns
                   if c.startswith(("psnr", "ssim", "mae", "input_psnr", "input_ssim", "input_mae", "det_"))]
    summary = stratified_summary(df, metric_cols)
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "stratified_metrics.csv"), index=False)
    for attr in ATTRIBUTES:
        s = summary[summary["attribute"] == attr]
        if len(s):
            s.to_csv(os.path.join(out_dir, f"by_{attr}.csv"), index=False)
    with open(os.path.join(out_dir, "bin_edges.json"), "w") as f:
        json.dump(edges, f, indent=2)
    if len(summary):
        plot_stratified(summary, os.path.join(out_dir, "plots"))
    plot_distributions(df, os.path.join(out_dir, "plots"))
    return df, summary, edges

"""Evaluation loop (used for validation during training and for the final test run)."""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .analysis import ATTRIBUTES, image_attributes, run_analysis
from .data import build_loader, scene_id
from .metrics import compute_metrics, detection_metrics
from .models import ShadowRemovalNet
from .utils import autocast, error_heatmap, load_checkpoint, save_comparison, save_image, to_uint8


def comparison_tiles(shadow, mask, out, gt, pred_mask=None):
    tiles = [("Shadow input", to_uint8(shadow)), ("Shadow mask (GT)", to_uint8(mask))]
    if pred_mask is not None:
        tiles.append(("Predicted mask", to_uint8(pred_mask)))
    tiles += [("Output", to_uint8(out)), ("Ground truth", to_uint8(gt)),
              ("|Error| x4", error_heatmap(out, gt))]
    return tiles


def _forward(model, batch, device, mask_source, amp):
    shadow = batch["shadow"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    with autocast(device, *amp):
        out = model(shadow, mask, mask_source)
    out = {k: v.float() for k, v in out.items()}
    return shadow, mask, gt, out


@torch.no_grad()
def evaluate(model, loader, device, mask_source="gt", amp=(False, torch.float32), save_dir=None,
             num_vis=0, vis_dir=None, compute_attrs=False, desc="eval"):
    """Returns (per-image DataFrame, mean inference ms/image)."""
    model.eval()
    rows, times, n_vis = [], [], 0
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        shadow, mask, gt, out = _forward(model, batch, device, mask_source, amp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) / shadow.shape[0])

        pred = out["output"].clamp(0, 1)
        met = compute_metrics(pred, gt, mask)
        base = compute_metrics(shadow, gt, mask)  # "do nothing" baseline
        det = detection_metrics(out["pred_mask"], mask) if "pred_mask" in out else {}

        for i in range(shadow.shape[0]):
            name = batch["name"][i]
            row = {"name": name, "index": int(batch["index"][i]), "scene": scene_id(name)}
            row.update({k: float(v[i]) for k, v in met.items()})
            row.update({f"input_{k}": float(v[i]) for k, v in base.items()})
            row.update({k: float(v[i]) for k, v in det.items()})
            if compute_attrs:
                row.update(image_attributes(shadow[i], gt[i], mask[i]))
            rows.append(row)

            if save_dir:
                save_image(pred[i], os.path.join(save_dir, "outputs", f"{name}.png"))
                if "pred_mask" in out:
                    save_image(out["pred_mask"][i], os.path.join(save_dir, "pred_masks", f"{name}.png"))
            if vis_dir and n_vis < num_vis:
                pm = out["pred_mask"][i] if "pred_mask" in out else None
                save_comparison(comparison_tiles(shadow[i], mask[i], pred[i], gt[i], pm),
                                os.path.join(vis_dir, f"{name}.png"))
                n_vis += 1
    # first iteration includes cuDNN autotuning; exclude it from timing when possible
    t = times[1:] if len(times) > 1 else times
    return pd.DataFrame(rows), 1000.0 * float(np.mean(t)) if t else float("nan")


def summarize(df):
    num = df.select_dtypes(include=[np.number]).drop(columns=["index"], errors="ignore")
    attr_cols = [c for c in ATTRIBUTES if c in num]
    return {k: float(v) for k, v in num.drop(columns=attr_cols).mean(skipna=True).items()}


def load_model_from_checkpoint(path, device):
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = ckpt["config"]
    model = ShadowRemovalNet.from_config(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg, ckpt


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def write_report(path, summary, strat, edges, mask_source, split, extra=None):
    regions = [("all", "Whole image"), ("shadow", "Shadow region"), ("nonshadow", "Non-shadow region")]
    lines = [f"# Shadow removal results — {split} split (mask source: `{mask_source}`)", ""]
    if extra:
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines += ["## Overall", "",
              "| Region | PSNR ↑ (input → model) | SSIM ↑ (input → model) | MAE LAB ↓ (input → model) "
              "| MAE RGB ↓ (input → model) |",
              "|---|---|---|---|---|"]
    for r, lab in regions:
        cells = []
        for m, nd in (("psnr", 2), ("ssim", 4), ("mae_lab", 2), ("mae_rgb", 2)):
            cells.append(f"{_fmt(summary.get(f'input_{m}_{r}'), nd)} → **{_fmt(summary.get(f'{m}_{r}'), nd)}**")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    if "det_ber" in summary:
        lines += ["", "## Shadow detection (detector branch)", "",
                  "| BER (%) ↓ | IoU ↑ | F1 ↑ |", "|---|---|---|",
                  f"| {_fmt(summary['det_ber'], 2)} | {_fmt(summary['det_iou'], 4)} | {_fmt(summary['det_f1'], 4)} |"]
    lines += ["", "## By shadow intensity and scene condition", "",
              "Bins are tertiles of each attribute over this split (edges in `bin_edges.json`).", ""]
    for attr in ATTRIBUTES:
        s = strat[strat["attribute"] == attr] if len(strat) else strat
        if not len(s):
            continue
        e = edges.get(attr)
        lines += [f"### {attr.replace('_', ' ').capitalize()}"
                  + (f" (edges: {', '.join(f'{x:.3f}' for x in e)})" if e else ""), "",
                  "| Bin | n | PSNR in → out | SSIM in → out | MAE LAB in → out | PSNR shadow out |",
                  "|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(
                f"| {r['bin']} | {int(r['n'])} | {_fmt(r.get('input_psnr_all'), 2)} → {_fmt(r.get('psnr_all'), 2)} "
                f"| {_fmt(r.get('input_ssim_all'), 4)} → {_fmt(r.get('ssim_all'), 4)} "
                f"| {_fmt(r.get('input_mae_lab_all'), 2)} → {_fmt(r.get('mae_lab_all'), 2)} "
                f"| {_fmt(r.get('psnr_shadow'), 2)} |")
        lines += ["", f"![{attr}](analysis/plots/by_{attr}.png)", ""]
    lines += ["## Distribution", "", "![psnr](analysis/plots/psnr_distribution.png)", "",
              "## Files", "",
              "- `per_image_metrics.csv` — every metric for every image (+ attributes)",
              "- `per_scene_metrics.csv` — metrics averaged per ISTD scene",
              "- `analysis/` — stratified tables and plots",
              "- `comparisons/`, `extremes/` — visual comparisons (first N, best/worst by PSNR)",
              "- `outputs/` — restored images (if `--save-images`)"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


@torch.no_grad()
def run_test(model, cfg, dataset, device, out_dir, mask_source="gt", amp=(False, torch.float32),
             save_images=True, num_vis=20, num_extremes=5, logger=None, split="test"):
    """Full evaluation + analysis + report. Returns the summary dict."""
    log = logger.info if logger else print
    os.makedirs(out_dir, exist_ok=True)
    loader = build_loader(dataset, batch_size=1, shuffle=False,
                          num_workers=cfg["data"].get("num_workers", 4), seed=cfg["experiment"]["seed"])
    df, ms = evaluate(model, loader, device, mask_source, amp,
                      save_dir=out_dir if save_images else None, num_vis=num_vis,
                      vis_dir=os.path.join(out_dir, "comparisons"), compute_attrs=True,
                      desc=f"{split}[{mask_source}]")

    df, strat, edges = run_analysis(df, os.path.join(out_dir, "analysis"),
                                    cfg.get("analysis", {}).get("bin_edges"))
    df.to_csv(os.path.join(out_dir, "per_image_metrics.csv"), index=False)
    num_cols = df.select_dtypes(include=[np.number]).columns.drop("index", errors="ignore")
    per_scene = df.groupby("scene")[list(num_cols)].mean()
    per_scene.insert(0, "n", df.groupby("scene").size())
    per_scene.to_csv(os.path.join(out_dir, "per_scene_metrics.csv"))

    summary = summarize(df)
    summary.update({"num_images": int(len(df)), "inference_ms_per_image": ms,
                    "mask_source": mask_source, "split": split,
                    "psnr_gain_all": summary["psnr_all"] - summary["input_psnr_all"]})

    # best / worst cases by whole-image PSNR
    order = df.sort_values("psnr_all")
    ext_dir = os.path.join(out_dir, "extremes")
    picks = [("worst", i, r) for i, (_, r) in enumerate(order.head(num_extremes).iterrows())] + \
            [("best", i, r) for i, (_, r) in enumerate(order.tail(num_extremes)[::-1].iterrows())]
    for tag, rank, r in picks:
        b = dataset[int(r["index"])]
        batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in b.items()}
        shadow, mask, gt, out = _forward(model, batch, device, mask_source, amp)
        pm = out["pred_mask"][0] if "pred_mask" in out else None
        save_comparison(comparison_tiles(shadow[0], mask[0], out["output"][0].clamp(0, 1), gt[0], pm),
                        os.path.join(ext_dir, f"{tag}_{rank + 1:02d}_{r['name']}_"
                                              f"{r['psnr_all']:.2f}dB.png"))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_report(os.path.join(out_dir, "report.md"), summary, strat, edges, mask_source, split,
                 extra={"Images": len(df), "Inference": f"{ms:.1f} ms / image ({device})"})

    log(f"[{split} | mask={mask_source}] PSNR {summary['psnr_all']:.2f} dB "
        f"(input {summary['input_psnr_all']:.2f}) | SSIM {summary['ssim_all']:.4f} | "
        f"MAE-LAB {summary['mae_lab_all']:.2f} | shadow PSNR {summary['psnr_shadow']:.2f} | "
        f"non-shadow PSNR {summary['psnr_nonshadow']:.2f}")
    log(f"Results written to {out_dir}")
    return summary

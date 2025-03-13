"""Evaluate a trained checkpoint on the ISTD test (or val) split.

    python test.py --checkpoint outputs/<run>/checkpoints/best.pth --gpu 0
    python test.py --checkpoint best.pth --gpu 1 --split val --mask-source pred
    python test.py --checkpoint best.pth --gpu 0 --opts data.eval_size=[256,256]   # 256x256 protocol
    python test.py --data-dir /mnt/shadow --checkpoint outputs/<run>/checkpoints/best.pth --gpu 0

Writes, per mask source, to <out-dir>/<split>_<source>/:
    summary.json, report.md, per_image_metrics.csv, per_scene_metrics.csv,
    analysis/ (shadow-intensity & scene-condition breakdown + plots),
    comparisons/, extremes/, outputs/, pred_masks/
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shadow_removal.data import build_datasets  # noqa: E402
from shadow_removal.evaluation import load_model_from_checkpoint, run_test  # noqa: E402
from shadow_removal.utils import (add_data_dir_arg, add_device_args, apply_data_dir,  # noqa: E402
                                  apply_overrides, describe_device, get_logger, load_config,
                                  resolve_amp, resolve_path, select_device, set_seed, setup_data_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="optional config (default: the one stored in the checkpoint)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--mask-source", default="both", choices=["gt", "pred", "both"])
    p.add_argument("--out-dir", default=None,
                   help="default: <run dir>/results (inside --data-dir when one is given)")
    p.add_argument("--no-save-images", action="store_true")
    p.add_argument("--num-vis", type=int, default=20)
    p.add_argument("--amp", default=None, choices=["none", "fp16", "bf16"], help="default: as trained")
    p.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. data.root=/data/ISTD")
    add_device_args(p)
    add_data_dir_arg(p)
    args = p.parse_args()

    data_root = setup_data_dir(args.data_dir)
    args.checkpoint = resolve_path(args.checkpoint, data_root)
    device = select_device(args.gpu)
    model, ckpt_cfg, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    if args.config:
        cfg = load_config(args.config, args.opts, data_root)
    else:
        cfg = apply_overrides(apply_data_dir(ckpt_cfg, data_root), args.opts)
    set_seed(cfg["experiment"]["seed"])

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    if args.out_dir:
        out_root = args.out_dir
    elif data_root and os.path.commonpath([run_dir, data_root]) != data_root:
        # checkpoint lives outside the data dir -> keep results inside it anyway
        out_root = os.path.join(data_root, "outputs", os.path.basename(run_dir), "results")
    else:
        out_root = os.path.join(run_dir, "results")
    os.makedirs(out_root, exist_ok=True)
    log = get_logger(os.path.join(out_root, "test.log"), name="shadow_removal.test").info
    amp = resolve_amp(device, args.amp or cfg["train"].get("amp", "none"))
    log(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')}) | device {describe_device(device)}")

    _, val_ds, test_ds = build_datasets(cfg)
    ds = test_ds if args.split == "test" else val_ds
    log(f"Evaluating {len(ds)} images from the {args.split} split")

    sources = ["gt", "pred"] if args.mask_source == "both" else [args.mask_source]
    if not model.has_detector and "pred" in sources:
        log("Model has no detector branch; evaluating with GT masks only")
        sources = ["gt"]

    results = {}
    for src in sources:
        results[src] = run_test(model, cfg, ds, device, os.path.join(out_root, f"{args.split}_{src}"),
                                mask_source=src, amp=amp, save_images=not args.no_save_images,
                                num_vis=args.num_vis, num_extremes=cfg.get("test", {}).get("num_extremes", 5),
                                logger=log, split=args.split)
    pd.DataFrame(results).T.to_csv(os.path.join(out_root, f"{args.split}_summary.csv"))
    log(f"Summary table: {os.path.join(out_root, f'{args.split}_summary.csv')}")


if __name__ == "__main__":
    main()

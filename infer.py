"""Remove shadows from arbitrary images (no ground truth needed).

    # with masks (file names must match the images)
    python infer.py --checkpoint best.pth --input photos/ --masks masks/ --output restored/ --gpu 0
    # without masks: the detector branch predicts them
    python infer.py --checkpoint best.pth --input photo.jpg --output restored/ --gpu 0
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shadow_removal.data import IMG_EXTS, _to_tensor  # noqa: E402
from shadow_removal.evaluation import load_model_from_checkpoint  # noqa: E402
from shadow_removal.utils import (add_device_args, autocast, resolve_amp, save_comparison,  # noqa: E402
                                  save_image, select_device, to_uint8)


def list_images(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in IMG_EXTS)


def find_mask(mask_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--masks", default=None, help="folder of binary shadow masks (optional)")
    p.add_argument("--output", default="inference_results")
    p.add_argument("--max-side", type=int, default=0, help="downscale so the longer side <= this (0 = off)")
    p.add_argument("--save-comparison", action="store_true")
    add_device_args(p)
    args = p.parse_args()

    device = select_device(args.gpu)
    model, cfg, _ = load_model_from_checkpoint(args.checkpoint, device)
    amp = resolve_amp(device, cfg["train"].get("amp", "none"))
    os.makedirs(args.output, exist_ok=True)

    for path in tqdm(list_images(args.input), desc="infer"):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        if args.max_side and max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BICUBIC)
        x = _to_tensor(img)[None].to(device)

        mask, src = None, "pred"
        mp = find_mask(args.masks, stem) if args.masks else None
        if mp:
            m = Image.open(mp).convert("L").resize(img.size, Image.NEAREST)
            mask = (_to_tensor(m)[None] > 0.5).float().to(device)
            src = "gt"
        elif not model.has_detector:
            print(f"skip {path}: no mask found and model has no detector")
            continue

        with autocast(device, *amp):
            out = model(x, mask, src)
        y = out["output"].float().clamp(0, 1)[0]
        save_image(y, os.path.join(args.output, f"{stem}.png"))
        guide = out["guide_mask"].float()[0]
        save_image(guide, os.path.join(args.output, f"{stem}_mask.png"))
        if args.save_comparison:
            save_comparison([("Input", to_uint8(x[0])), ("Mask", to_uint8(guide)), ("Output", to_uint8(y))],
                            os.path.join(args.output, f"{stem}_comparison.png"))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

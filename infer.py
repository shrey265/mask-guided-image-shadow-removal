"""Remove shadows from arbitrary images (no ground truth needed).

    # with masks (file names must match the images)
    python infer.py --checkpoint best.pth --input photos/ --masks masks/ --output restored/ --gpu 0
    # without masks: the detector branch predicts them
    python infer.py --checkpoint best.pth --input photo.jpg --output restored/ --gpu 0
    # everything under one folder (checkpoint path may be relative to it; output -> <dir>/inference)
    python infer.py --data-dir /mnt/shadow --checkpoint outputs/<run>/checkpoints/best.pth --input photos/
"""
import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shadow_removal.data import IMG_EXTS  # noqa: E402
from shadow_removal.inference import ShadowRemover, load_image  # noqa: E402
from shadow_removal.utils import (add_data_dir_arg, add_device_args, resolve_path,  # noqa: E402
                                  save_comparison, save_image, select_device, setup_data_dir, to_uint8)


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--masks", default=None, help="folder of binary shadow masks (optional)")
    p.add_argument("--output", default=None,
                   help="default: <data-dir>/inference when --data-dir is given, else ./inference_results")
    p.add_argument("--max-side", type=int, default=0, help="downscale so the longer side <= this (0 = off)")
    p.add_argument("--save-comparison", action="store_true")
    add_device_args(p)
    add_data_dir_arg(p)
    args = p.parse_args()

    data_root = setup_data_dir(args.data_dir)
    args.checkpoint = resolve_path(args.checkpoint, data_root)
    if args.output is None:
        args.output = os.path.join(data_root, "inference") if data_root else "inference_results"
    remover = ShadowRemover(args.checkpoint, select_device(args.gpu))
    os.makedirs(args.output, exist_ok=True)

    for path in tqdm(list_images(args.input), desc="infer"):
        stem = os.path.splitext(os.path.basename(path))[0]
        mp = find_mask(args.masks, stem) if args.masks else None
        if mp is None and not remover.has_detector:
            print(f"skip {path}: no mask found and model has no detector")
            continue
        r = remover(load_image(path), load_image(mp, "L") if mp else None, args.max_side)
        save_image(r["output"], os.path.join(args.output, f"{stem}.png"))
        save_image(r["mask"], os.path.join(args.output, f"{stem}_mask.png"))
        if args.save_comparison:
            save_comparison([("Input", to_uint8(r["input"])), ("Mask", to_uint8(r["mask"])),
                             ("Output", to_uint8(r["output"]))],
                            os.path.join(args.output, f"{stem}_comparison.png"))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

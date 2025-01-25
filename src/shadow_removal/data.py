"""ISTD-style shadow removal dataset: (shadow image, shadow mask, shadow-free image) triplets.

Expected layout (ISTD):
    <root>/train/train_A  shadow images
    <root>/train/train_B  binary shadow masks
    <root>/train/train_C  shadow-free ground truth
    <root>/test/test_A ... test/test_C

Directory names are configurable (``{split}`` is substituted), so ISTD+ (adjusted
ground truth, e.g. ``train_C_fixed_ours``) or SRD-style folders also work.
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _index_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    out = {}
    for f in sorted(os.listdir(path)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS:
            out[stem] = os.path.join(path, f)
    return out


def find_triplets(data_cfg, split):
    """Return a sorted list of {name, shadow, mask, gt} dicts for ``split`` ('train'/'test')."""
    base = os.path.join(data_cfg["root"], split)
    dirs = {k: os.path.join(base, data_cfg[f"{k}_dir"].format(split=split))
            for k in ("shadow", "mask", "gt")}
    idx = {k: _index_dir(v) for k, v in dirs.items()}
    common = sorted(set(idx["shadow"]) & set(idx["mask"]) & set(idx["gt"]))
    missing = len(idx["shadow"]) - len(common)
    if not common:
        raise RuntimeError(f"No matching triplets found under {base} (dirs: {dirs})")
    if missing > 0:
        print(f"[data] warning: {missing} shadow images in '{split}' lack a mask or GT; skipped")
    return [{"name": n, "shadow": idx["shadow"][n], "mask": idx["mask"][n], "gt": idx["gt"][n]}
            for n in common]


def scene_id(name):
    """ISTD names are '<scene>-<view>' (e.g. '90-3'); images of one scene share a background."""
    return name.split("-")[0] if "-" in name else name


def split_train_val(samples, val_ratio, seed=42, by_scene=True):
    """Hold out a validation set from the training split.

    With ``by_scene`` the split is done on scene ids, so no background appears in both
    train and val (ISTD has many views of the same scene; a per-image split leaks).
    """
    if val_ratio <= 0:
        return samples, []
    rng = random.Random(seed)
    if by_scene:
        groups = defaultdict(list)
        for s in samples:
            groups[scene_id(s["name"])].append(s)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, int(round(len(keys) * val_ratio)))
        val_keys = set(keys[:n_val])
        train = [s for s in samples if scene_id(s["name"]) not in val_keys]
        val = [s for s in samples if scene_id(s["name"]) in val_keys]
    else:
        order = list(range(len(samples)))
        rng.shuffle(order)
        n_val = max(1, int(round(len(samples) * val_ratio)))
        val_idx = set(order[:n_val])
        train = [s for i, s in enumerate(samples) if i not in val_idx]
        val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def _to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


class ShadowDataset(Dataset):
    """Loads aligned triplets. Training: random crop + flips. Eval: full image (or ``eval_size``)."""

    def __init__(self, samples, train=False, crop_size=256, load_size=None, eval_size=None,
                 hflip=True, vflip=False, mask_threshold=0.5):
        self.samples = samples
        self.train = train
        self.crop_size = crop_size
        self.load_size = load_size
        self.eval_size = eval_size
        self.hflip = hflip
        self.vflip = vflip
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resize(img, size_hw, resample):
        h, w = size_hw
        return img if img.size == (w, h) else img.resize((w, h), resample)

    def __getitem__(self, i):
        s = self.samples[i]
        shadow = Image.open(s["shadow"]).convert("RGB")
        mask = Image.open(s["mask"]).convert("L")
        gt = Image.open(s["gt"]).convert("RGB")

        size = self.load_size if self.train else self.eval_size
        target = tuple(size) if size else (shadow.size[1], shadow.size[0])
        shadow = self._resize(shadow, target, Image.BICUBIC)
        gt = self._resize(gt, target, Image.BICUBIC)
        mask = self._resize(mask, target, Image.NEAREST)

        shadow, mask, gt = _to_tensor(shadow), _to_tensor(mask), _to_tensor(gt)

        if self.train:
            shadow, mask, gt = self._augment(shadow, mask, gt)

        mask = (mask > self.mask_threshold).float()
        return {"shadow": shadow, "mask": mask, "gt": gt, "name": s["name"], "index": i}

    def _augment(self, shadow, mask, gt):
        c = self.crop_size
        if c:
            _, h, w = shadow.shape
            if h < c or w < c:  # upscale small images so a crop fits
                scale = c / min(h, w)
                nh, nw = int(np.ceil(h * scale)), int(np.ceil(w * scale))
                shadow = torch.nn.functional.interpolate(shadow[None], (nh, nw), mode="bicubic",
                                                         align_corners=False)[0].clamp(0, 1)
                gt = torch.nn.functional.interpolate(gt[None], (nh, nw), mode="bicubic",
                                                     align_corners=False)[0].clamp(0, 1)
                mask = torch.nn.functional.interpolate(mask[None], (nh, nw), mode="nearest")[0]
                h, w = nh, nw
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            sl = (slice(None), slice(top, top + c), slice(left, left + c))
            shadow, mask, gt = shadow[sl], mask[sl], gt[sl]
        if self.hflip and random.random() < 0.5:
            shadow, mask, gt = shadow.flip(-1), mask.flip(-1), gt.flip(-1)
        if self.vflip and random.random() < 0.5:
            shadow, mask, gt = shadow.flip(-2), mask.flip(-2), gt.flip(-2)
        return shadow.contiguous(), mask.contiguous(), gt.contiguous()


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_datasets(cfg):
    """Returns (train_ds, val_ds, test_ds) with a scene-disjoint validation split."""
    d = cfg["data"]
    seed = cfg["experiment"]["seed"]
    all_train = find_triplets(d, d["train_split"])
    train_s, val_s = split_train_val(all_train, d["val_ratio"], seed, d.get("split_by_scene", True))
    test_s = find_triplets(d, d["test_split"])
    common = dict(eval_size=d.get("eval_size"), mask_threshold=d.get("mask_threshold", 0.5))
    train_ds = ShadowDataset(train_s, train=True, crop_size=d["crop_size"], load_size=d.get("load_size"),
                             hflip=d.get("hflip", True), vflip=d.get("vflip", False), **common)
    val_ds = ShadowDataset(val_s, train=False, **common)
    test_ds = ShadowDataset(test_s, train=False, **common)
    return train_ds, val_ds, test_ds


def build_loader(ds, batch_size, shuffle, num_workers, seed=42, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      worker_init_fn=_worker_init, generator=g,
                      persistent_workers=num_workers > 0)

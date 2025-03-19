"""Single-image inference shared by infer.py and the web app."""
import threading
import time

import torch
from PIL import Image, ImageOps

from .data import _to_tensor
from .evaluation import load_model_from_checkpoint
from .metrics import compute_metrics
from .utils import autocast, resolve_amp


def load_image(source, mode="RGB"):
    """Open a path / file object, apply EXIF rotation (phone photos), convert mode."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert(mode)


def fit_max_side(img, max_side):
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s))), Image.BICUBIC)
    return img


class ShadowRemover:
    """Loads a checkpoint once; thread-safe __call__ (one GPU forward at a time)."""

    def __init__(self, checkpoint, device, amp=None):
        self.model, self.cfg, ckpt = load_model_from_checkpoint(checkpoint, device)
        self.device = device
        self.amp = resolve_amp(device, amp or self.cfg["train"].get("amp", "none"))
        self.epoch = ckpt.get("epoch")
        self.best_val = ckpt.get("best_val")
        self.best_metric = ckpt.get("best_metric")
        self._lock = threading.Lock()

    @property
    def has_detector(self):
        return self.model.has_detector

    @torch.no_grad()
    def __call__(self, image, mask=None, max_side=0):
        """
        image: PIL RGB image; mask: optional PIL image (white = shadow).
        Returns dict of CPU tensors (input, output, mask) + mask_source + time_ms.
        """
        image = fit_max_side(image.convert("RGB"), max_side)
        x = _to_tensor(image)[None]
        m, src = None, "pred"
        if mask is not None:
            mask = mask.convert("L").resize(image.size, Image.NEAREST)
            m = (_to_tensor(mask)[None] > 0.5).float()
            src = "gt"
        elif not self.has_detector:
            raise ValueError("This model has no shadow detector, so a mask must be provided")

        with self._lock:
            x_d = x.to(self.device)
            m_d = m.to(self.device) if m is not None else None
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            with autocast(self.device, *self.amp):
                out = self.model(x_d, m_d, src)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ms = 1000.0 * (time.perf_counter() - t0)
            result = {
                "input": x[0],
                "output": out["output"].float().clamp(0, 1)[0].cpu(),
                "mask": out["guide_mask"].float()[0].cpu(),
                "mask_source": src,
                "time_ms": ms,
                "size": image.size,
            }
        return result


@torch.no_grad()
def score_against_gt(result, gt_image, gt_mask):
    """PSNR/SSIM/MAE of model output and of the raw input vs. a ground truth (ISTD examples)."""
    w, h = result["size"]
    gt = _to_tensor(gt_image.convert("RGB").resize((w, h), Image.BICUBIC))[None]
    gm = (_to_tensor(gt_mask.convert("L").resize((w, h), Image.NEAREST))[None] > 0.5).float()
    model = compute_metrics(result["output"][None], gt, gm)
    base = compute_metrics(result["input"][None], gt, gm)
    return {k: float(v[0]) for k, v in model.items()}, {k: float(v[0]) for k, v in base.items()}

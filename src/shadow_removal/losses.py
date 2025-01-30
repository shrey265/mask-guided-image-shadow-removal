"""Training objective for mask-guided shadow removal.

L = w_l1 * L1(out, gt)
  + w_shadow_l1 * L1 restricted to the shadow region      (focus on the hard pixels)
  + w_ssim * (1 - SSIM)                                   (structure)
  + w_gradient * L1(grad(out), grad(gt))                  (texture / edges)
  + w_perceptual * VGG16 feature L1                       (texture, optional)
  + w_mask_bce * BCE(mask_logits, mask) + w_mask_dice * Dice   (detector, if enabled)
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ssim


def _image_gradients(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG16 relu1_2 / relu2_2 / relu3_3 features (ImageNet weights)."""

    LAYERS = (3, 8, 15)

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        try:  # torchvision >= 0.13
            from torchvision.models import VGG16_Weights
            net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except ImportError:  # pragma: no cover - very old torchvision
            net = vgg16(pretrained=True)
        self.features = net.features[: max(self.LAYERS) + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):  # frozen feature extractor stays in eval mode
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.features):
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + F.l1_loss(x, y)
        return loss / len(self.LAYERS)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class ShadowRemovalLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w = {k: float(v) for k, v in weights.items()}
        self.perceptual = None
        if self.w.get("perceptual", 0) > 0:
            try:
                self.perceptual = VGGPerceptualLoss()
            except Exception as e:  # e.g. no internet to fetch VGG weights
                warnings.warn(f"Perceptual loss disabled (could not load VGG16 weights: {e})")
                self.w["perceptual"] = 0.0

    def forward(self, out, batch):
        pred = out["output"].float()
        gt = batch["gt"].float()
        mask = batch["mask"].float()
        parts = {}

        if self.w.get("l1", 0) > 0:
            parts["l1"] = F.l1_loss(pred, gt)
        if self.w.get("shadow_l1", 0) > 0:
            m = mask.expand_as(pred)
            parts["shadow_l1"] = ((pred - gt).abs() * m).sum() / m.sum().clamp_min(1.0)
        if self.w.get("ssim", 0) > 0:
            parts["ssim"] = 1.0 - ssim(pred, gt).mean()
        if self.w.get("gradient", 0) > 0:
            pdx, pdy = _image_gradients(pred)
            gdx, gdy = _image_gradients(gt)
            parts["gradient"] = F.l1_loss(pdx, gdx) + F.l1_loss(pdy, gdy)
        if self.perceptual is not None and self.w.get("perceptual", 0) > 0:
            parts["perceptual"] = self.perceptual(pred, gt)
        if "mask_logits" in out:
            logits = out["mask_logits"].float()
            if self.w.get("mask_bce", 0) > 0:
                parts["mask_bce"] = F.binary_cross_entropy_with_logits(logits, mask)
            if self.w.get("mask_dice", 0) > 0:
                parts["mask_dice"] = dice_loss(logits, mask)

        total = sum(self.w[k] * v for k, v in parts.items())
        logs = {k: v.detach().item() for k, v in parts.items()}
        logs["total"] = total.detach().item()
        return total, logs

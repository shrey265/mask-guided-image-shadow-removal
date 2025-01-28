"""Mask-guided U-Net for shadow removal.

ShadowRemovalNet = (optional) shadow detector U-Net  ->  mask-guided removal U-Net

* The removal U-Net takes RGB + shadow mask (4 channels) as input.
* Every skip connection passes through a MaskGuidedAttention gate that combines the
  encoder feature, the decoder gating signal and the (resized) shadow mask, so the
  decoder concentrates reconstruction capacity on shadow regions while leaving lit
  regions (and their structure/texture) untouched.
* The network predicts a residual that is added to the input (shadow removal is mostly
  a local, spatially-varying illumination correction).
* An optional light detector U-Net predicts the shadow mask from RGB, so the model can
  also be used on images without a ground-truth mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind, ch):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "in":
        return nn.InstanceNorm2d(ch, affine=True)
    if kind == "gn":
        return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)
    if kind in (None, "none"):
        return nn.Identity()
    raise ValueError(f"unknown norm '{kind}'")


class ConvBlock(nn.Module):
    """(conv3x3 -> norm -> ReLU) x 2"""

    def __init__(self, cin, cout, norm="bn", dropout=0.0):
        super().__init__()
        bias = norm in (None, "none")
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=bias), norm_layer(norm, cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=bias), norm_layer(norm, cout), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MaskGuidedAttention(nn.Module):
    """Attention gate on a skip connection, conditioned on the shadow mask.

    a = sigmoid(psi(relu(W_s * skip + W_g * gate + W_m * mask)))
    out = skip * (1 + a)       # emphasise shadow-relevant features, never fully block lit ones
    """

    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.w_mask = nn.Sequential(nn.Conv2d(1, inter_ch, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(inter_ch, inter_ch, 3, padding=1))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate, mask):
        m = F.interpolate(mask, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.w_skip(skip) + self.w_gate(gate) + self.w_mask(m))
        return skip * (1.0 + a)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64, depth=4, norm="bn", mask_guided=False, dropout=0.0):
        super().__init__()
        self.depth = depth
        self.mask_guided = mask_guided
        chs = [min(base * 2 ** i, base * 8) for i in range(depth + 1)]

        self.inc = ConvBlock(in_ch, chs[0], norm)
        self.downs = nn.ModuleList(
            [ConvBlock(chs[i], chs[i + 1], norm, dropout if i == depth - 1 else 0.0) for i in range(depth)])
        self.ups, self.atts, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.atts.append(MaskGuidedAttention(chs[i], chs[i], max(chs[i] // 2, 16)) if mask_guided
                             else nn.Identity())
            self.decs.append(ConvBlock(chs[i] * 2, chs[i], norm))
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x, mask=None):
        h_in, w_in = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (m - h_in % m) % m, (m - w_in % m) % m
        if ph or pw:  # arbitrary input sizes
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            if mask is not None:
                mask = F.pad(mask, (0, pw, 0, ph), mode="replicate")

        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(F.max_pool2d(feats[-1], 2)))
        h = feats.pop()
        for up, att, dec in zip(self.ups, self.atts, self.decs):
            h = up(h)
            skip = feats.pop()
            if self.mask_guided:
                skip = att(skip, h, mask)
            h = dec(torch.cat([h, skip], dim=1))
        return self.outc(h)[..., :h_in, :w_in]


class ShadowRemovalNet(nn.Module):
    def __init__(self, base_channels=64, depth=4, norm="bn", mask_guided_skips=True, residual=True,
                 composite_output=False, composite_dilate=7, dropout=0.0, detector=None):
        super().__init__()
        detector = detector or {}
        self.residual = residual
        self.composite_output = composite_output
        self.composite_dilate = composite_dilate | 1  # odd kernel
        self.remover = UNet(4, 3, base_channels, depth, norm, mask_guided=mask_guided_skips, dropout=dropout)
        self.detector = None
        if detector.get("enabled", False):
            self.detector = UNet(3, 1, detector.get("base_channels", 32), detector.get("depth", 4), norm)

    @classmethod
    def from_config(cls, model_cfg):
        return cls(**model_cfg)

    @property
    def has_detector(self):
        return self.detector is not None

    def _soft_region(self, mask):
        k = self.composite_dilate
        dil = F.max_pool2d(mask, k, stride=1, padding=k // 2)
        return F.avg_pool2d(dil, k, stride=1, padding=k // 2, count_include_pad=False)

    def forward(self, shadow, mask=None, mask_source="gt"):
        """
        shadow: (N,3,H,W) in [0,1]; mask: (N,1,H,W) in {0,1} or None
        mask_source: 'gt' uses the given mask, 'pred' uses the detector's prediction.
        """
        out = {}
        if self.detector is not None:
            logits = self.detector(shadow)
            out["mask_logits"] = logits
            out["pred_mask"] = torch.sigmoid(logits)

        if mask_source == "pred" or mask is None:
            if self.detector is None:
                raise ValueError("No mask given and the model has no shadow detector "
                                 "(set model.detector.enabled: true)")
            guide = out["pred_mask"]
        else:
            guide = mask

        r = self.remover(torch.cat([shadow, guide], dim=1), guide)
        img = shadow + r if self.residual else r
        if self.composite_output:
            soft = self._soft_region(guide)
            img = soft * img + (1.0 - soft) * shadow
        out["output"] = img
        out["guide_mask"] = guide
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

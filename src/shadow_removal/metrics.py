"""Restoration metrics (pure PyTorch, no extra dependencies).

All functions take batched tensors in [0,1]: pred/gt (N,3,H,W), mask (N,1,H,W).
Metrics are reported on three regions, as is standard for shadow removal:
    *_all       whole image
    *_shadow    pixels inside the ground-truth shadow mask
    *_nonshadow pixels outside it

* PSNR         peak signal-to-noise ratio (dB, data range 1.0)
* SSIM         Gaussian-window SSIM (11x11, sigma 1.5, K1=0.01, K2=0.03), region = masked mean of SSIM map
* MAE_LAB      mean absolute error in CIE-LAB (the "RMSE" column of the ISTD literature)
* MAE_RGB      mean absolute error in sRGB on the 0-255 scale
"""
import torch
import torch.nn.functional as F

_EPS = 1e-8


# ----------------------------------------------------------------------------- SSIM
def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_map(x, y, data_range=1.0, win_size=11, sigma=1.5):
    c = x.shape[1]
    w = _gaussian_window(win_size, sigma, c, x.device, x.dtype)
    pad = win_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim(x, y, data_range=1.0):
    return ssim_map(x, y, data_range).mean(dim=(1, 2, 3))


# ----------------------------------------------------------------------------- colour
def rgb_to_lab(rgb):
    """sRGB [0,1] -> CIE-LAB (D65). L in [0,100]."""
    rgb = rgb.clamp(0, 1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]], device=rgb.device, dtype=rgb.dtype)
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    white = torch.tensor([0.950456, 1.0, 1.088754], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    t = (xyz / white).clamp_min(1e-8)
    f = torch.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


# ----------------------------------------------------------------------------- helpers
def _masked_mean(v, m):
    """Mean of v (N,C,H,W) over pixels where m (N,1,H,W) == 1; NaN when the region is empty."""
    m = m.expand_as(v)
    num = (v * m).sum(dim=(1, 2, 3))
    den = m.sum(dim=(1, 2, 3))
    out = num / den.clamp_min(_EPS)
    return torch.where(den > 0, out, torch.full_like(out, float("nan")))


def _psnr_from_mse(mse):
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def psnr(pred, gt):
    return _psnr_from_mse(((pred - gt) ** 2).mean(dim=(1, 2, 3)))


@torch.no_grad()
def compute_metrics(pred, gt, mask):
    """Returns {metric_name: tensor(N)} for each image in the batch."""
    pred = pred.float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    mask = (mask.float() > 0.5).float()
    regions = {"all": torch.ones_like(mask), "shadow": mask, "nonshadow": 1.0 - mask}

    se = (pred - gt) ** 2
    smap = ssim_map(pred, gt)
    ae_rgb = (pred - gt).abs() * 255.0
    ae_lab = (rgb_to_lab(pred) - rgb_to_lab(gt)).abs()

    out = {}
    for r, m in regions.items():
        out[f"psnr_{r}"] = _psnr_from_mse(_masked_mean(se, m))
        out[f"ssim_{r}"] = _masked_mean(smap, m)
        out[f"mae_lab_{r}"] = _masked_mean(ae_lab, m)
        out[f"mae_rgb_{r}"] = _masked_mean(ae_rgb, m)
    return out


@torch.no_grad()
def detection_metrics(pred_prob, gt_mask, threshold=0.5):
    """Shadow detection quality: BER (balanced error rate, %), IoU, F1. Per image."""
    p = (pred_prob.float() > threshold).float()
    g = (gt_mask.float() > 0.5).float()
    dims = (1, 2, 3)
    tp = (p * g).sum(dims)
    tn = ((1 - p) * (1 - g)).sum(dims)
    fp = (p * (1 - g)).sum(dims)
    fn = ((1 - p) * g).sum(dims)
    pos_acc = tp / (tp + fn).clamp_min(_EPS)
    neg_acc = tn / (tn + fp).clamp_min(_EPS)
    ber = 100.0 * (1.0 - 0.5 * (pos_acc + neg_acc))
    iou = tp / (tp + fp + fn).clamp_min(_EPS)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(_EPS)
    return {"det_ber": ber, "det_iou": iou, "det_f1": f1}


# Metrics where lower is better (used for checkpoint selection & plots)
LOWER_IS_BETTER = ("mae", "ber", "loss")


def is_lower_better(name):
    return any(k in name for k in LOWER_IS_BETTER)

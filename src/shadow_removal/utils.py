"""Config, device selection, seeding, checkpoints, logging and image I/O."""
import contextlib
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------------- config
def load_config(path, overrides=None, data_dir=None):
    """Load YAML, point it at ``data_dir`` (if given), then apply dotted overrides,
    e.g. ['train.lr=1e-4', 'model.detector.enabled=false']. Overrides win over data_dir."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    apply_data_dir(cfg, data_dir)
    return apply_overrides(cfg, overrides)


# ----------------------------------------------------------------------------- data dir
DATA_DIR_ENV = "SHADOW_DATA_DIR"


def add_data_dir_arg(parser):
    parser.add_argument(
        "--data-dir", default=os.environ.get(DATA_DIR_ENV),
        help="single folder for everything the project writes: dataset (<dir>/ISTD_Dataset), "
             "runs/checkpoints/results (<dir>/outputs) and library caches such as the VGG16 weights "
             f"(<dir>/cache). Default: ${DATA_DIR_ENV}, else ./data and ./outputs")


def setup_data_dir(data_dir):
    """Create the data-dir layout and redirect library caches into it.

    Must run before torch.hub downloads (VGG16 weights) and before matplotlib is imported.
    Returns the absolute path, or None when no data dir was given.
    """
    if not data_dir:
        return None
    root = os.path.abspath(os.path.expanduser(data_dir))
    cache = os.path.join(root, "cache")
    for sub in ("outputs", "cache/torch/hub", "cache/matplotlib"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    os.environ["TORCH_HOME"] = os.path.join(cache, "torch")      # pretrained weights
    os.environ["MPLCONFIGDIR"] = os.path.join(cache, "matplotlib")  # matplotlib font cache
    os.environ["XDG_CACHE_HOME"] = cache                          # other well-behaved libraries
    torch.hub.set_dir(os.path.join(cache, "torch", "hub"))
    os.environ[DATA_DIR_ENV] = root                               # inherited by DataLoader workers
    return root


def apply_data_dir(cfg, root):
    """Point dataset + outputs of a config at ``root`` (no-op when root is None)."""
    if root:
        ds_name = os.path.basename(os.path.normpath(cfg["data"]["root"])) or "ISTD_Dataset"
        cfg["data"]["root"] = os.path.join(root, ds_name)
        cfg["experiment"]["output_dir"] = os.path.join(root, "outputs")
        cfg["experiment"]["data_dir"] = root
    return cfg


def resolve_path(path, root):
    """Allow paths relative to the data dir, e.g. --checkpoint outputs/<run>/checkpoints/best.pth."""
    if path and not os.path.exists(path) and root and os.path.exists(os.path.join(root, path)):
        return os.path.join(root, path)
    return path


def apply_overrides(cfg, overrides=None):
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must look like key.subkey=value")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    _coerce_numbers(cfg)
    return cfg


def _coerce_numbers(node):
    """PyYAML reads '2e-4' as a string; convert such values to float."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                _coerce_numbers(v)
            elif isinstance(v, str):
                try:
                    if any(c in v.lower() for c in "e.") and v.lower() not in ("none", "nan", "inf"):
                        node[k] = float(v)
                except ValueError:
                    pass
    elif isinstance(node, list):
        for v in node:
            _coerce_numbers(v)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ----------------------------------------------------------------------------- device
def select_device(gpu):
    """Pick exactly one device.

    gpu >= 0 -> cuda:<gpu> (index among the *visible* devices, i.e. after CUDA_VISIBLE_DEVICES)
    gpu <  0 -> CPU
    """
    if gpu is None or gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"--gpu {gpu} requested but CUDA is not available "
                           "(check `python check_env.py`, or use --gpu -1 for CPU)")
    n = torch.cuda.device_count()
    if gpu >= n:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        hint = f" (CUDA_VISIBLE_DEVICES={vis} remaps indices to 0..{n - 1})" if vis else ""
        raise ValueError(f"--gpu {gpu} requested but only {n} CUDA device(s) are visible{hint}")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def describe_device(device):
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{device} ({p.name}, {p.total_memory / 1024 ** 3:.1f} GB)"
    return "cpu"


def add_device_args(parser):
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index to use (single GPU). -1 = CPU. "
                             "Run `python check_env.py` to list devices.")


# ----------------------------------------------------------------------------- AMP
def resolve_amp(device, amp_cfg):
    """Returns (enabled, dtype). 'bf16' falls back to fp16 on GPUs without bf16 support."""
    if device.type != "cuda" or not amp_cfg or amp_cfg in ("none", False):
        return False, torch.float32
    if amp_cfg == "bf16":
        # native bf16 needs Ampere+ (cc >= 8.0); recent torch reports emulated (slow) bf16 on T4/V100
        if torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        amp_cfg = "fp16"
    if amp_cfg == "fp16":
        return True, torch.float16
    raise ValueError(f"amp must be one of none|fp16|bf16, got {amp_cfg}")


def autocast(device, enabled, dtype):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(enabled):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------------------- misc
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_logger(log_file=None, name="shadow_removal"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class MeterDict:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def update(self, values, n=1):
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * n
            self.counts[k] = self.counts.get(k, 0) + n

    def averages(self):
        return {k: self.sums[k] / max(self.counts[k], 1) for k in self.sums}


def save_checkpoint(path, **state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location=map_location)


# ----------------------------------------------------------------------------- images
def to_uint8(t):
    """(C,H,W) tensor in [0,1] -> (H,W,3) uint8."""
    t = t.detach().float().clamp(0, 1).cpu()
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def save_image(t, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(to_uint8(t)).save(path)


def error_heatmap(pred, gt, gain=4.0):
    """Per-pixel absolute error as a colour heatmap (H,W,3) uint8 (magma-like, dark = low error)."""
    import matplotlib
    err = ((pred.float() - gt.float()).abs().mean(0) * gain).clamp(0, 1).cpu().numpy()
    cmap = matplotlib.colormaps["magma"] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap("magma")
    return (cmap(err)[..., :3] * 255).astype(np.uint8)


def save_comparison(tiles, path, label_h=22):
    """tiles: list of (label, HxWx3 uint8). Saves a labelled horizontal strip."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h = max(t.shape[0] for _, t in tiles)
    w = sum(t.shape[1] for _, t in tiles)
    canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, arr in tiles:
        canvas.paste(Image.fromarray(arr), (x, label_h))
        draw.text((x + 4, 4), label, fill=(20, 20, 20))
        x += arr.shape[1]
    canvas.save(path)

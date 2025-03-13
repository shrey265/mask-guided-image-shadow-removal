"""Train the mask-guided shadow removal U-Net.

    python train.py --config configs/istd.yaml --gpu 0
    python train.py --config configs/istd.yaml --gpu 3 --opts train.batch_size=16 train.epochs=300
    python train.py --resume outputs/<run>/checkpoints/last.pth --gpu 0
    python train.py --data-dir /mnt/shadow --gpu 0     # dataset, runs, checkpoints, caches -> /mnt/shadow
"""
import argparse
import math
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from shadow_removal.data import build_datasets, build_loader  # noqa: E402
from shadow_removal.evaluation import evaluate, run_test, summarize  # noqa: E402
from shadow_removal.losses import ShadowRemovalLoss  # noqa: E402
from shadow_removal.metrics import is_lower_better  # noqa: E402
from shadow_removal.models import ShadowRemovalNet, count_parameters  # noqa: E402
from shadow_removal.utils import (MeterDict, add_data_dir_arg, add_device_args, apply_data_dir,  # noqa: E402
                                  autocast, describe_device, get_logger, load_checkpoint, load_config,
                                  make_grad_scaler, resolve_amp, resolve_path, save_checkpoint,
                                  save_config, select_device, set_seed, setup_data_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/istd.yaml")
    p.add_argument("--resume", default=None, help="checkpoint to resume from (uses its run directory)")
    p.add_argument("--exp-name", default=None, help="override experiment.name")
    p.add_argument("--opts", nargs="*", default=[], help="config overrides: key.sub=value")
    add_device_args(p)
    add_data_dir_arg(p)
    return p.parse_args()


def build_optimizer(model, tcfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)  # no weight decay on norms / biases
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    cls = torch.optim.AdamW if tcfg.get("optimizer", "adamw") == "adamw" else torch.optim.Adam
    return cls(groups, lr=tcfg["lr"], betas=tuple(tcfg.get("betas", (0.9, 0.999))))


def build_scheduler(opt, tcfg, iters_per_epoch):
    total = max(1, tcfg["epochs"] * iters_per_epoch)
    warm = int(tcfg.get("warmup_epochs", 0) * iters_per_epoch)
    floor = tcfg.get("min_lr_ratio", 0.01)

    def fn(it):
        if it < warm:
            return (it + 1) / warm
        prog = min(1.0, (it - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def plot_history(hist_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shadow_removal.analysis import COLOR_INPUT, COLOR_MODEL, GRID, TEXT_PRIMARY, TEXT_SECONDARY

    df = pd.read_csv(hist_csv)
    panels = [("train_total", "Training loss", COLOR_MODEL),
              ("val_psnr_all", "Validation PSNR (dB) ↑", COLOR_MODEL),
              ("val_ssim_all", "Validation SSIM ↑", COLOR_MODEL),
              ("val_mae_lab_all", "Validation MAE LAB ↓", COLOR_MODEL),
              ("val_psnr_shadow", "Validation PSNR, shadow region ↑", COLOR_MODEL),
              ("lr", "Learning rate", COLOR_INPUT)]
    panels = [p for p in panels if p[0] in df and df[p[0]].notna().any()]
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, (col, title, color) in zip(axes.flat, panels):
        d = df[["epoch", col]].dropna()
        ax.plot(d["epoch"], d[col], color=color, linewidth=2, marker="o" if len(d) < 60 else None, markersize=4)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        if col == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data_root = setup_data_dir(args.data_dir)
    resume_ckpt = None
    if args.resume:
        args.resume = resolve_path(args.resume, data_root)
        resume_ckpt = load_checkpoint(args.resume)
        cfg = apply_data_dir(resume_ckpt["config"], data_root)
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
    else:
        cfg = load_config(args.config, args.opts, data_root)
        if args.exp_name:
            cfg["experiment"]["name"] = args.exp_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg["experiment"]["output_dir"], f"{cfg['experiment']['name']}_{stamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    log = get_logger(os.path.join(run_dir, "train.log")).info

    tcfg, dcfg = cfg["train"], cfg["data"]
    set_seed(cfg["experiment"]["seed"], cfg["experiment"].get("deterministic", False))
    device = select_device(args.gpu)
    amp = resolve_amp(device, tcfg.get("amp", "none"))
    log(f"Run dir: {run_dir}")
    log(f"Dataset: {cfg['data']['root']}" + (f" | data dir: {data_root}" if data_root else ""))
    log(f"Device: {describe_device(device)} | torch {torch.__version__} | AMP: {amp[0]} {amp[1]}")

    # ---------------------------------------------------------------- data
    train_ds, val_ds, test_ds = build_datasets(cfg)
    log(f"Data: train {len(train_ds)} | val {len(val_ds)} (scene-disjoint={dcfg.get('split_by_scene', True)}) "
        f"| test {len(test_ds)}")
    with open(os.path.join(run_dir, "splits.txt"), "w") as f:
        for tag, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            f.writelines(f"{tag}\t{s['name']}\n" for s in ds.samples)
    train_loader = build_loader(train_ds, tcfg["batch_size"], True, dcfg["num_workers"],
                                cfg["experiment"]["seed"], drop_last=True)
    val_loader = build_loader(val_ds, 1, False, dcfg["num_workers"]) if len(val_ds) else None

    # ---------------------------------------------------------------- model
    model = ShadowRemovalNet.from_config(cfg["model"]).to(device)
    n_rem = count_parameters(model.remover)
    n_det = count_parameters(model.detector) if model.has_detector else 0
    log(f"Model: removal U-Net {n_rem / 1e6:.2f}M params | detector {n_det / 1e6:.2f}M params")
    criterion = ShadowRemovalLoss(cfg["loss"]).to(device)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg, len(train_loader))
    scaler = make_grad_scaler(amp[0] and amp[1] == torch.float16)

    best_metric = tcfg.get("best_metric", "psnr_all")
    minimize = is_lower_better(best_metric)
    best_val = float("inf") if minimize else -float("inf")
    start_epoch, history, bad_vals = 1, [], 0
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        if resume_ckpt.get("scaler_state"):
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt.get("best_val", best_val)
        hist_path = os.path.join(run_dir, "history.csv")
        if os.path.exists(hist_path):
            history = pd.read_csv(hist_path).to_dict("records")
        log(f"Resumed from {args.resume} at epoch {start_epoch}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    except Exception:
        tb = None

    def state(epoch):
        return dict(epoch=epoch, config=cfg, model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                    scaler_state=scaler.state_dict(), best_val=best_val, best_metric=best_metric)

    # ---------------------------------------------------------------- loop
    pred_prob = tcfg.get("pred_mask_prob", 0.0) if model.has_detector else 0.0
    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        meter = MeterDict()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            shadow = batch["shadow"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            batch_dev = {"gt": batch["gt"].to(device, non_blocking=True), "mask": mask}
            src = "pred" if random.random() < pred_prob else "gt"

            with autocast(device, *amp):
                out = model(shadow, mask, src)
            loss, parts = criterion(out, batch_dev)

            if not torch.isfinite(loss):
                log(f"Non-finite loss at epoch {epoch} iter {it} ({parts}); skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            meter.update(parts, shadow.shape[0])

            if tb is not None and global_step % 10 == 0:
                for k, v in parts.items():
                    tb.add_scalar(f"train/{k}", v, global_step)
            if (it + 1) % tcfg.get("log_every", 50) == 0:
                log(f"ep {epoch:03d} it {it + 1:04d}/{len(train_loader)} | "
                    + " ".join(f"{k} {v:.4f}" for k, v in parts.items())
                    + f" | lr {optimizer.param_groups[0]['lr']:.2e}")

        tr = meter.averages()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0}
        row.update({f"train_{k}": v for k, v in tr.items()})
        log(f"Epoch {epoch:03d} done in {row['time_s']:.0f}s | loss {tr.get('total', float('nan')):.4f}")

        # ---------------------------------------------------------- validation
        is_val = val_loader is not None and (epoch % tcfg["val_every"] == 0 or epoch == tcfg["epochs"])
        if is_val:
            vis_dir = os.path.join(run_dir, "val_vis", f"epoch_{epoch:03d}")
            vdf, _ = evaluate(model, val_loader, device, tcfg.get("val_mask_source", "gt"), amp,
                              num_vis=tcfg.get("num_val_vis", 4), vis_dir=vis_dir, desc="val")
            vs = summarize(vdf)
            row.update({f"val_{k}": v for k, v in vs.items()})
            cur = vs[best_metric]
            improved = cur < best_val if minimize else cur > best_val
            msg = (f"  val PSNR {vs['psnr_all']:.2f} (shadow {vs['psnr_shadow']:.2f}, non-shadow "
                   f"{vs['psnr_nonshadow']:.2f}) | SSIM {vs['ssim_all']:.4f} | MAE-LAB {vs['mae_lab_all']:.3f}")
            if "det_ber" in vs:
                msg += f" | det BER {vs['det_ber']:.2f}"
            if tb is not None:
                for k, v in vs.items():
                    tb.add_scalar(f"val/{k}", v, epoch)
            if improved:
                best_val, bad_vals = cur, 0
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), **state(epoch))
                msg += f"  <- new best {best_metric}={cur:.4f}"
            else:
                bad_vals += 1
            log(msg)

        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        save_checkpoint(os.path.join(ckpt_dir, "last.pth"), **state(epoch))
        if tcfg.get("save_every", 0) and epoch % tcfg["save_every"] == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"), **state(epoch))

        patience = tcfg.get("early_stop_patience", 0)
        if patience and bad_vals >= patience:
            log(f"Early stopping: no improvement in {best_metric} for {patience} validations")
            break

    if tb is not None:
        tb.close()
    try:
        plot_history(os.path.join(run_dir, "history.csv"), os.path.join(run_dir, "training_curves.png"))
    except Exception as e:  # plotting must never kill a finished run
        log(f"Could not plot training curves: {e}")

    best_path = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(best_path):  # no validation set -> use last
        best_path = os.path.join(ckpt_dir, "last.pth")
    log(f"Training finished. Best {best_metric}: {best_val:.4f} | checkpoint: {best_path}")

    # ---------------------------------------------------------------- test
    if tcfg.get("run_test_after_train", True):
        model.load_state_dict(load_checkpoint(best_path)["model_state"])
        test_cfg = cfg.get("test", {})
        sources = [s for s in test_cfg.get("mask_sources", ["gt"]) if s == "gt" or model.has_detector]
        all_res = {}
        for src in sources:
            all_res[src] = run_test(model, cfg, test_ds, device, os.path.join(run_dir, "results", f"test_{src}"),
                                    mask_source=src, amp=amp, save_images=test_cfg.get("save_images", True),
                                    num_vis=test_cfg.get("num_vis", 20),
                                    num_extremes=test_cfg.get("num_extremes", 5), logger=log)
        pd.DataFrame(all_res).T.to_csv(os.path.join(run_dir, "results", "test_summary.csv"))


if __name__ == "__main__":
    main()

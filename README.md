# Mask-Guided Shadow Removal for Natural Scene Images

A U-Net encoder-decoder that removes shadows from natural scene photos. The RGB image and its shadow mask go in together, so the network rebuilds the shadow region and leaves the rest of the scene's structure and texture alone. A small detector branch predicts the shadow mask, so the model also works on photos that come without one.

Trained and evaluated on **ISTD** (Wang et al., CVPR 2018), using **PSNR, SSIM and MAE**. Scores are reported on the whole image, the shadow region and the non-shadow region. They are also broken down by **shadow intensity** and **scene condition** (shadow area, scene brightness, texture).

---

## 1. Method

```
                  ┌──────────────────────── Detector U-Net (32-ch, optional) ───────────┐
 shadow RGB ──────┤  RGB ─► mask logits ─► sigmoid ─► predicted mask                    │
     │            └─────────────────────────────────────────────────┬──────────────────┘
     │                                     GT mask (train/test) ──┐ │ (inference w/o mask)
     │                                                            ▼ ▼
     │      ┌──────────────── Mask-guided removal U-Net (64-ch, depth 4) ───────────────┐
     └─────►│ [RGB ‖ mask] (4ch) ─► Enc1 ─► Enc2 ─► Enc3 ─► Enc4 ─► Bottleneck          │
            │                        │       │       │       │          │              │
            │      skip ─► MaskGuidedAttention(skip, decoder gate, mask) ─► Dec ...     │
            │                                                     ─► residual (3ch)    │
            └───────────────────────────────────────────────────────────────┬──────────┘
                                                 output = shadow RGB + residual
```

| Component | Details |
|---|---|
| Input | RGB + binary shadow mask concatenated (4 channels) |
| Encoder / decoder | 4 levels of (Conv3×3-BN-ReLU)×2, channels 64→128→256→512→512, transposed-conv upsampling |
| **Mask-guided skips** | Each skip feature is gated by `a = σ(ψ(W_s·skip + W_g·gate + W_m·mask))` and becomes `skip·(1+a)`. The gate puts more weight on shadow-region features without blocking lit-region detail. |
| Residual output | The network predicts a correction that is added to the input. Shadow removal is a local illumination change, so predicting the correction preserves texture. |
| Detector branch | A lighter U-Net (RGB→mask) trained with BCE + Dice loss. It provides the mask at inference time. During training, 20% of batches are guided by its mask (`pred_mask_prob`), so the removal network learns to tolerate imperfect masks. |
| Arbitrary sizes | Inputs are padded internally to a multiple of 16, so full 640×480 images and any other size work. |

**Loss**: `L1 + shadow-region L1 + 0.5·(1−SSIM) + 0.2·gradient-L1 + 0.05·VGG16 perceptual + 0.5·(BCE + Dice)` for the mask. Every weight can be changed in `configs/istd.yaml`.

**Training**: AdamW (lr 2e-4, wd 1e-4), linear warmup then cosine decay, bf16/fp16 AMP, gradient clipping, 256×256 random crops with horizontal flips, 200 epochs, batch size 8.

## 2. Dataset: ISTD

| Split | Triplets | Use |
|---|---|---|
| `train` | 1,330 | ~90% for training, ~10% for **validation**. The validation part is split by **scene** (ISTD file names are `<scene>-<view>`), so the same background never appears in both. |
| `test` | 540 | Final test, never seen during training or model selection |

Each triplet is `*_A` (shadow image), `*_B` (shadow mask) and `*_C` (shadow-free ground truth). The exact train/val/test file lists are written to `splits.txt` in each run folder.

> To use **ISTD+** (the colour-adjusted ground truth of Le & Samaras 2019), set `data.gt_dir="{split}_C_fixed_ours"`.

```bash
bash scripts/download_istd.sh                       # -> data/ISTD_Dataset (needs 7z / unrar)
bash scripts/download_istd.sh /path/ISTD_Dataset.rar  # or point it at an archive you already have
```

If the Google Drive download fails because of quota limits, download the archive by hand from the [ST-CGAN repo](https://github.com/DeepInsight-PCALab/ST-CGAN) and pass it to the script.

## 3. Installation (no torch version conflicts)

`torch` and `torchvision` are **installed separately, as a pinned matching pair** (2.5.1 / 0.20.1) from the official PyTorch wheel index. Nothing in `requirements.txt` depends on torch, and `constraints.txt` pins the pair, so a later `pip install` cannot swap in a mismatched torch.

```bash
nvidia-smi                      # check your driver version
bash setup_env.sh cu121         # driver >= 530   (default)
bash setup_env.sh cu124         # driver >= 550
bash setup_env.sh cu118         # older drivers (>= 520)
source .venv/bin/activate
```

The script creates `.venv`, removes any old torch/torchvision, installs the matching pair, installs the other dependencies under the constraints file, and then runs `check_env.py`. That check verifies:

- torch and torchvision are a matching pair
- numpy and torch work together
- each GPU can run a forward and backward pass
- torchvision's CUDA ops load (they fail if torchvision was built against a different torch)

Manual install:

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt
python check_env.py
```

Python 3.9–3.12. The wheels ship their own CUDA runtime and cuDNN, so the server only needs a recent enough driver, not a CUDA toolkit.

## 4. Choosing one GPU on a multi-GPU server

Every script runs on **exactly one** device, chosen with `--gpu`:

```bash
python check_env.py                 # lists all visible GPUs (name, memory, bf16 support)
python train.py --gpu 3             # train on cuda:3
python test.py  --gpu 1 ...         # evaluate on cuda:1
python train.py --gpu -1            # CPU (debug only)
```

`--gpu` counts among the *visible* devices. If you also set `CUDA_VISIBLE_DEVICES=2,3`, then `--gpu 0` means physical GPU 2. An invalid index stops with a clear error instead of silently falling back to another device.

## 5. Training (train + validation)

```bash
python train.py --config configs/istd.yaml --gpu 0
python train.py --gpu 0 --opts train.batch_size=16 train.epochs=300 model.detector.enabled=false
python train.py --resume outputs/<run>/checkpoints/last.pth --gpu 0      # resume
tensorboard --logdir outputs/                                            # optional
```

Validation runs every `val_every` epochs at full resolution. The checkpoint with the best `train.best_metric` (default: whole-image val PSNR) is saved as `best.pth`. When training ends, **the test set is evaluated automatically** with `best.pth`, once with the GT mask and once with the predicted mask.

Or run everything end to end: `bash scripts/run_pipeline.sh <gpu_id>`.

## 6. Testing / inference

```bash
python test.py --checkpoint outputs/<run>/checkpoints/best.pth --gpu 0                    # GT + predicted masks
python test.py --checkpoint ... --gpu 0 --mask-source gt --opts data.eval_size=[256,256]  # 256×256 protocol
python test.py --checkpoint ... --gpu 0 --split val

python infer.py --checkpoint best.pth --input my_photos/ --output restored/ --gpu 0 --save-comparison
python infer.py --checkpoint best.pth --input my_photos/ --masks my_masks/ --output restored/ --gpu 0
```

## 7. Saved results

```
outputs/<experiment>_<timestamp>/
├── config.yaml                  exact config used
├── splits.txt                   train / val / test file lists
├── train.log                    full training log
├── history.csv                  per-epoch train losses, lr, val metrics
├── training_curves.png          loss / val PSNR / SSIM / MAE / lr curves
├── tensorboard/
├── val_vis/epoch_XXX/           validation comparisons during training
├── checkpoints/{best,last,epoch_XXX}.pth
└── results/
    ├── test_summary.csv         one row per mask source (gt / pred)
    └── test_gt/ , test_pred/
        ├── report.md            human-readable results report with tables and plots
        ├── summary.json         all averaged metrics + inference time
        ├── per_image_metrics.csv
        ├── per_scene_metrics.csv
        ├── analysis/
        │   ├── by_shadow_intensity.csv, by_shadow_area.csv,
        │   │   by_scene_brightness.csv, by_scene_texture.csv, stratified_metrics.csv
        │   ├── bin_edges.json
        │   └── plots/by_*.png, psnr_distribution.png
        ├── comparisons/         input | GT mask | pred mask | output | GT | error map
        ├── extremes/            5 best and 5 worst images by PSNR
        ├── outputs/             every restored test image
        └── pred_masks/          detector masks
```

## 8. Metrics

| Metric | Definition |
|---|---|
| **PSNR** (dB, ↑) | `10·log10(1/MSE)` on [0,1] RGB, per image, then averaged |
| **SSIM** (↑) | Gaussian 11×11 window, σ=1.5. Region SSIM is the masked mean of the SSIM map. |
| **MAE LAB** (↓) | Mean absolute error in CIE-LAB. This is the "RMSE" column in most ISTD papers. |
| **MAE RGB** (↓) | Mean absolute error on the 0–255 scale |
| Detection | BER (balanced error rate, %), IoU and F1 for the predicted masks |

Every metric is reported on the **whole image**, the **shadow region** and the **non-shadow region**. Each one is shown next to the **input baseline**, i.e. the shadow image scored as-is against the ground truth, so the gain from the model is explicit.

### Shadow intensity and scene-condition analysis

For each test image the following attributes are computed from the ground truth, so they don't depend on the model:

| Attribute | Definition | Bins (tertiles) |
|---|---|---|
| Shadow intensity | `1 − L̄(shadow img ∩ mask) / L̄(shadow-free ∩ mask)` | light / medium / strong |
| Shadow area | fraction of pixels inside the mask | small / medium / large |
| Scene brightness | mean luminance of the shadow-free image | dark / medium / bright |
| Scene texture | mean gradient magnitude of the shadow-free image | smooth / medium / textured |

PSNR, SSIM and MAE are averaged per bin for the model and the input baseline. They are saved as CSV files and bar charts, and summarised in `report.md`. To use fixed bin edges instead of tertiles, set `analysis.bin_edges`.

## 9. Results

After training, copy the numbers from `results/test_gt/report.md`:

| Region | PSNR ↑ | SSIM ↑ | MAE LAB ↓ |
|---|---|---|---|
| Whole image | – | – | – |
| Shadow | – | – | – |
| Non-shadow | – | – | – |

## 10. Project layout

```
├── configs/istd.yaml            all hyper-parameters
├── src/shadow_removal/
│   ├── data.py                  ISTD loader, scene-disjoint val split, augmentation
│   ├── models.py                mask-guided U-Net, attention gates, detector
│   ├── losses.py                L1 / shadow-L1 / SSIM / gradient / VGG / mask losses
│   ├── metrics.py               PSNR, SSIM, MAE (RGB & LAB), detection metrics
│   ├── analysis.py              shadow-intensity / scene-condition analysis and plots
│   ├── evaluation.py            evaluation loop, report writer
│   └── utils.py                 config, single-GPU selection, AMP, checkpoints, images
├── train.py  test.py  infer.py  check_env.py
├── scripts/download_istd.sh  scripts/run_pipeline.sh
├── setup_env.sh  requirements.txt  requirements-torch.txt  constraints.txt
```

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: operator torchvision::nms does not exist` | torch and torchvision don't match. Re-run `bash setup_env.sh <cuda>`. |
| `CUDA driver version is insufficient` | Use an older wheel index (`cu118`) or update the driver. |
| CUDA out of memory | Lower `train.batch_size` or `data.crop_size`, or set `model.base_channels=32`. |
| No internet on the server, VGG weights can't download | Set `loss.perceptual=0`. The run continues anyway and prints a warning. |
| fp16 loss becomes NaN | Use `train.amp=bf16` (Ampere or newer) or `train.amp=none`. |

## References

- J. Wang, X. Li, J. Yang. *Stacked Conditional GANs for Jointly Learning Shadow Detection and Shadow Removal*, CVPR 2018 (ISTD).
- O. Ronneberger et al. *U-Net*, MICCAI 2015. O. Oktay et al. *Attention U-Net*, MIDL 2018.
- H. Le, D. Samaras. *Shadow Removal via Shadow Image Decomposition*, ICCV 2019 (ISTD+).

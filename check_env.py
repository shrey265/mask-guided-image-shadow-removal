"""Sanity-check the Python / PyTorch / CUDA environment before training.

    python check_env.py            # checks every visible GPU
    python check_env.py --gpu 2    # checks only cuda:2
"""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="only test this CUDA device index")
    args = parser.parse_args()

    ok = True
    print(f"Python      : {sys.version.split()[0]}")
    try:
        import torch
        import torchvision
    except ImportError as e:
        print(f"[FAIL] cannot import torch/torchvision: {e}")
        print("       Run: bash setup_env.sh <cu118|cu121|cu124>")
        sys.exit(1)

    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")

    # torch/torchvision pairing check (torchvision's C++ ops are compiled per torch release)
    expected = {"2.1": "0.16", "2.2": "0.17", "2.3": "0.18", "2.4": "0.19",
                "2.5": "0.20", "2.6": "0.21", "2.7": "0.22", "2.8": "0.23"}
    t_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    tv_mm = ".".join(torchvision.__version__.split("+")[0].split(".")[:2])
    if t_mm in expected and expected[t_mm] != tv_mm:
        print(f"[FAIL] torch {t_mm} requires torchvision {expected[t_mm]}.x, found {tv_mm}")
        ok = False
    else:
        print("[ OK ] torch / torchvision versions are a matched pair")

    for pkg in ["numpy", "PIL", "yaml", "tqdm", "pandas", "matplotlib"]:
        try:
            mod = importlib.import_module(pkg)
            print(f"[ OK ] {pkg:<11} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"[FAIL] {pkg} missing (pip install -r requirements.txt -c constraints.txt)")
            ok = False
    try:
        import tensorboard  # noqa: F401
        print("[ OK ] tensorboard")
    except ImportError:
        print("[WARN] tensorboard missing - TensorBoard logging will be skipped")

    # numpy <-> torch interop (catches numpy ABI mismatches)
    import numpy as np
    try:
        torch.from_numpy(np.ones(3, dtype=np.float32)).numpy()
        print("[ OK ] numpy <-> torch interop")
    except Exception as e:  # pragma: no cover
        print(f"[FAIL] numpy <-> torch interop: {e}")
        ok = False

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available - training will run on CPU (very slow)")
    else:
        n = torch.cuda.device_count()
        print(f"Visible GPUs: {n}")
        ids = [args.gpu] if args.gpu is not None else list(range(n))
        for i in ids:
            if i < 0 or i >= n:
                print(f"[FAIL] cuda:{i} does not exist (visible devices: 0..{n - 1})")
                ok = False
                continue
            p = torch.cuda.get_device_properties(i)
            try:
                dev = torch.device(f"cuda:{i}")
                x = torch.randn(2, 4, 64, 64, device=dev)
                conv = torch.nn.Conv2d(4, 8, 3, padding=1).to(dev)
                y = conv(x).sum()
                y.backward()
                # torchvision CUDA op - fails loudly if torchvision was built for another torch
                boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float, device=dev)
                torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=dev), 0.5)
                bf16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
                print(f"[ OK ] cuda:{i} {p.name} | {p.total_memory / 1024**3:.1f} GB | "
                      f"cc {p.major}.{p.minor} | bf16={bf16}")
            except Exception as e:
                print(f"[FAIL] cuda:{i} {p.name}: {e}")
                ok = False

    print("\nEnvironment OK" if ok else "\nEnvironment has problems (see [FAIL] lines)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

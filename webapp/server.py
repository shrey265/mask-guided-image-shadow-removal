"""Local web UI for the shadow removal model.

Runs on the server; open it from your laptop through an SSH tunnel:

    # server
    python webapp/server.py --data-dir /mnt/shadow --gpu 2 \
        --checkpoint outputs/<run>/checkpoints/best.pth
    # laptop
    ssh -L 8000:localhost:8000 user@server      ->  http://localhost:8000

By default it listens on 127.0.0.1 only (not reachable from the network). Uploaded images
are processed in memory and discarded unless --save is given.
"""
import argparse
import base64
import io
import math
import os
import re
import secrets
import sys
from datetime import datetime
from functools import lru_cache
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "webapp", "static")
sys.path.insert(0, os.path.join(ROOT, "src"))

from PIL import Image  # noqa: E402

from shadow_removal.data import find_triplets  # noqa: E402
from shadow_removal.inference import ShadowRemover, load_image, score_against_gt  # noqa: E402
from shadow_removal.utils import (add_data_dir_arg, add_device_args, apply_data_dir,  # noqa: E402
                                  describe_device, resolve_path, save_image, select_device,
                                  setup_data_dir, to_uint8)

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="trained model (path may be relative to --data-dir)")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 = only via SSH tunnel (default). 0.0.0.0 = whole network, token-protected")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token", default=None, help="access token (auto-generated when --host is not local)")
    p.add_argument("--max-side", type=int, default=2048,
                   help="hard cap on the longer image side processed (protects GPU memory)")
    p.add_argument("--max-upload-mb", type=float, default=25)
    p.add_argument("--save", action="store_true",
                   help="keep uploads + results in <data-dir>/webapp/ (default: discard)")
    p.add_argument("--istd-root", default=None, help="ISTD folder for the examples tab (default: from config)")
    add_device_args(p)
    add_data_dir_arg(p)
    return p.parse_args()


# ----------------------------------------------------------------------------- encoding helpers
def _encode(t_or_img, fmt="PNG", quality=92):
    img = t_or_img if isinstance(t_or_img, Image.Image) else Image.fromarray(to_uint8(t_or_img))
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def data_url(t_or_img, fmt="PNG"):
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(_encode(t_or_img, fmt)).decode("ascii")


def _clean(d):
    """JSON cannot carry NaN (e.g. shadow-region metrics on an image with an empty mask)."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(name or "image")[0])[:60] or "image"


# ----------------------------------------------------------------------------- app
def create_app(remover, examples, opts):
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Shadow Removal", docs_url=None, redoc_url=None, openapi_url=None)
    token = opts["token"]
    max_bytes = int(opts["max_upload_mb"] * 1024 * 1024)
    names = sorted(examples)

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        if token and request.url.path.startswith("/api/"):
            given = request.headers.get("x-token") or request.query_params.get("token") or ""
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "Missing or invalid access token"}, status_code=401)
        return await call_next(request)

    def effective_side(requested):
        cap = opts["max_side"]
        return min(requested, cap) if requested and requested > 0 else cap

    def read_upload(upload, what, mode):
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"{what} is larger than {opts['max_upload_mb']:g} MB")
        if not data:
            raise HTTPException(400, f"{what} is empty")
        try:
            return load_image(io.BytesIO(data), mode)
        except Exception:
            raise HTTPException(400, f"{what} is not a readable image file")

    def pack(r):
        w, h = r["size"]
        return {"input": data_url(r["input"], "JPEG"), "output": data_url(r["output"]),
                "mask": data_url(r["mask"]), "mask_source": r["mask_source"],
                "time_ms": round(r["time_ms"], 1), "width": w, "height": h}

    def maybe_save(tag, r, extra=None):
        if not opts["save_dir"]:
            return None
        folder = os.path.join(opts["save_dir"], f"{datetime.now():%Y%m%d_%H%M%S_%f}_{tag}")
        save_image(r["input"], os.path.join(folder, "input.png"))
        save_image(r["output"], os.path.join(folder, "output.png"))
        save_image(r["mask"], os.path.join(folder, f"mask_{r['mask_source']}.png"))
        for k, img in (extra or {}).items():
            img.save(os.path.join(folder, f"{k}.png"))
        return folder

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})

    @app.get("/api/info")
    def info():
        return _clean({
            "checkpoint": opts["checkpoint_label"], "epoch": remover.epoch,
            "best_metric": remover.best_metric,
            "best_val": float(remover.best_val) if remover.best_val is not None else None,
            "device": describe_device(remover.device), "amp": str(remover.amp[1]).replace("torch.", "")
            if remover.amp[0] else "fp32",
            "has_detector": remover.has_detector, "num_examples": len(names),
            "examples_note": opts["examples_note"], "max_side": opts["max_side"],
            "max_upload_mb": opts["max_upload_mb"], "saving": bool(opts["save_dir"]),
        })

    @app.get("/api/examples")
    def list_examples(q: str = "", offset: int = 0, limit: int = 60):
        hits = [n for n in names if q.strip().lower() in n.lower()]
        limit = max(1, min(limit, 200))
        return {"total": len(hits), "items": hits[max(0, offset):max(0, offset) + limit]}

    @lru_cache(maxsize=1024)
    def thumb_bytes(name):
        img = load_image(examples[name]["shadow"])
        img.thumbnail((240, 240))
        return _encode(img, "JPEG", 85)

    @app.get("/api/examples/{name}/thumb")
    def example_thumb(name: str):
        if name not in examples:
            raise HTTPException(404, "Unknown example")
        return Response(thumb_bytes(name), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/remove")
    def remove(image: UploadFile = File(...), mask: Optional[UploadFile] = File(None),
               max_side: int = Form(1024)):
        img = read_upload(image, "Image", "RGB")
        m = read_upload(mask, "Mask", "L") if mask is not None and mask.filename else None
        if m is None and not remover.has_detector:
            raise HTTPException(400, "This model has no shadow detector, so please upload a mask")
        r = remover(img, m, effective_side(max_side))
        out = pack(r)
        out["saved_to"] = maybe_save(_safe_name(image.filename), r)
        return out

    @app.post("/api/examples/{name}/run")
    def run_example(name: str, mask_source: str = "gt", max_side: int = 0):
        if name not in examples:
            raise HTTPException(404, "Unknown example")
        if mask_source not in ("gt", "pred"):
            raise HTTPException(400, "mask_source must be 'gt' or 'pred'")
        if mask_source == "pred" and not remover.has_detector:
            raise HTTPException(400, "This model has no shadow detector")
        s = examples[name]
        shadow, gt_mask, gt = load_image(s["shadow"]), load_image(s["mask"], "L"), load_image(s["gt"])
        r = remover(shadow, gt_mask if mask_source == "gt" else None, effective_side(max_side))
        model_m, input_m = score_against_gt(r, gt, gt_mask)
        out = pack(r)
        w, h = r["size"]
        out.update({
            "name": name,
            "gt": data_url(gt.resize((w, h), Image.BICUBIC), "JPEG"),
            "gt_mask": data_url(gt_mask.resize((w, h), Image.NEAREST)),
            "metrics": {"model": _clean(model_m), "input": _clean(input_m)},
            "saved_to": maybe_save(f"istd_{_safe_name(name)}", r),
        })
        return out

    return app


def load_examples(remover, data_root, istd_root):
    """ISTD test triplets for the examples tab ({} + note if the dataset is not available)."""
    import copy
    cfg = apply_data_dir(copy.deepcopy(remover.cfg), data_root)
    dcfg = cfg["data"]
    if istd_root:
        dcfg["root"] = istd_root
    try:
        samples = find_triplets(dcfg, dcfg["test_split"])
        return {s["name"]: s for s in samples}, f"ISTD test set at {dcfg['root']}"
    except Exception as e:
        return {}, f"ISTD examples unavailable ({e}). Pass --data-dir or --istd-root."


def main():
    args = parse_args()
    try:
        import uvicorn
        import fastapi  # noqa: F401
        import multipart  # noqa: F401  (python-multipart, needed for uploads)
    except ImportError as e:
        sys.exit(f"Missing web dependency ({e.name}). Install with:\n"
                 "  python -m pip install -r requirements-web.txt -c constraints.txt")

    data_root = setup_data_dir(args.data_dir)
    ckpt = resolve_path(args.checkpoint, data_root)
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    device = select_device(args.gpu)
    print(f"Loading {ckpt} on {describe_device(device)} ...")
    remover = ShadowRemover(ckpt, device)
    examples, note = load_examples(remover, data_root, args.istd_root)
    print(note)

    local = args.host in LOCAL_HOSTS
    token = args.token or (None if local else secrets.token_urlsafe(16))
    save_dir = None
    if args.save:
        save_dir = os.path.join(data_root, "webapp") if data_root else os.path.join(ROOT, "webapp_results")
        os.makedirs(save_dir, exist_ok=True)

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    opts = {"token": token, "max_side": args.max_side, "max_upload_mb": args.max_upload_mb,
            "save_dir": save_dir, "examples_note": note,
            "checkpoint_label": f"{os.path.basename(run_dir)}/{os.path.basename(ckpt)}"}
    app = create_app(remover, examples, opts)

    q = f"/?token={token}" if token else "/"
    print("\n" + "=" * 70)
    if local:
        print(f"Web UI listening on http://{args.host}:{args.port} (server-local only)")
        print("On your LAPTOP run:")
        print(f"    ssh -N -L {args.port}:localhost:{args.port} <user>@<this-server>")
        print(f"then open  http://localhost:{args.port}{q}")
    else:
        print(f"Web UI listening on http://{args.host}:{args.port} (reachable from the network!)")
        print(f"Open  http://<server-ip>:{args.port}{q}")
    print("Uploads are " + (f"saved to {save_dir}" if save_dir else "processed in memory and discarded"))
    print("=" * 70 + "\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

import os, sys, json, traceback, base64, uuid, time
import runpod
from urllib.request import urlopen, Request
from app.inference import run_inference_animatediff

TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def _save_tmp_image_from_url(url: str) -> str:
    print(f"[HANDLER] downloading image_url: {url}", flush=True)
    req = Request(url, headers={"User-Agent": "curl/8"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
    fname = f"in_{uuid.uuid4().hex}.bin"
    out_path = os.path.join(TMP_DIR, fname)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path

def _save_tmp_image_from_b64(b64: str) -> str:
    print("[HANDLER] got image_base64", flush=True)
    data = base64.b64decode(b64)
    fname = f"in_{uuid.uuid4().hex}.bin"
    out_path = os.path.join(TMP_DIR, fname)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path

def handler(event):
    t0 = time.time()
    try:
        payload = event.get("input") or {}
        image_url = payload.get("image_url")
        image_b64 = payload.get("image_base64")
        seed = int(payload.get("seed", 1234))
        num_frames = int(payload.get("num_frames", 16))
        fps = int(payload.get("fps", 8))
        guidance_scale = float(payload.get("guidance_scale", 1.0))

        if image_url:
            image_path = _save_tmp_image_from_url(image_url)
        elif image_b64:
            image_path = _save_tmp_image_from_b64(image_b64)
        else:
            raise ValueError("image_url or image_base64 required")

        # --- 핵심: 반환값 2튜플/3튜플 모두 안전하게 처리 ---
        res = run_inference_animatediff(
            image_path=image_path,
            seed=seed,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
        )
        if isinstance(res, tuple):
            if len(res) == 3:
                ok, out_video, reason = res
            elif len(res) == 2:
                ok, out_video = res
                reason = None
            else:
                ok, out_video, reason = False, None, f"unexpected return shape: {len(res)}"
        else:
            ok, out_video, reason = False, None, "unexpected return type"

        ms = int((time.time() - t0) * 1000)

        if not ok:
            return {
                "success": False,
                "error": "inference_failed",
                "reason": reason,
                "output": {"ms": ms},
            }

        return {"success": True, "output": {"video_path": out_video, "ms": ms}}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

runpod.serverless.start({"handler": handler})

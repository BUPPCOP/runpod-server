import os, sys, json, traceback, base64, uuid, time
import runpod
from urllib.request import urlopen, Request
from app.inference import run_inference_animatediff

TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def _save_input_image(image_url: str | None, image_base64: str | None) -> str:
    """image_url 또는 image_base64 중 하나를 받아 /tmp에 저장하고 경로 반환."""
    out_path = os.path.join(TMP_DIR, f"input_{uuid.uuid4().hex}.jpg")
    if image_url:
        print(f"[HANDLER] downloading image_url: {image_url}", flush=True)
        # 간단/안전: urllib 직접 사용 (추가 deps 불필요)
        req = Request(image_url, headers={"User-Agent": "curl/8"})
        with urlopen(req, timeout=30) as r, open(out_path, "wb") as f:
            f.write(r.read())
        return out_path
    if image_base64:
        print("[HANDLER] decoding image_base64", flush=True)
        data = base64.b64decode(image_base64)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path
    raise ValueError("image_url or image_base64 is required")

def handler(event):
    t0 = time.time()
    try:
        print("[HANDLER] event received:", json.dumps({
            "id": event.get("id"),
            "has_input": bool(event.get("input")),
        }), flush=True)

        payload = event.get("input") or {}
        image_url = payload.get("image_url")
        image_b64 = payload.get("image_base64") or payload.get("image_b64")  # 두 키 모두 허용
        seed = payload.get("seed")
        num_frames = int(payload.get("num_frames", 16))
        fps = int(payload.get("fps", 8))
        guidance_scale = float(payload.get("guidance_scale", 1.0))

        # 입력 확보
        img_path = _save_input_image(image_url, image_b64)
        out_path = os.path.join(TMP_DIR, f"out_{uuid.uuid4().hex}.mp4")

        # 추론
        ok = run_inference_animatediff(
            image_path=img_path,
            output_path=out_path,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        elapsed = int((time.time() - t0) * 1000)
        if not ok:
            print("[HANDLER] inference_failed", flush=True)
            return {"error": "inference_failed", "ms": elapsed}

        print(f"[HANDLER] done -> {out_path}", flush=True)
        return {
            "success": True,
            "ms": elapsed,
            "video_path": out_path  # 필요하면 presigned URL 업로드로 교체 가능
        }

    except Exception as e:
        print("[HANDLER][EXC]", repr(e), flush=True)
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

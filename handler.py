# handler.py
import os
import base64
import tempfile
from pathlib import Path

import requests
import runpod

from app.inference import run_inference_animatediff

OUT_DIR = Path("/app/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _download_to_tmp(url_or_b64: str) -> str:
    """image_url(HTTP) 또는 base64 문자열을 임시 파일로 저장 후 경로 반환"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
        r = requests.get(url_or_b64, timeout=60)
        r.raise_for_status()
        tmp.write(r.content)
    else:
        # base64: data URL 형식도 허용 (data:image/png;base64,....)
        if url_or_b64.startswith("data:"):
            url_or_b64 = url_or_b64.split(",", 1)[-1]
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def handler(job):
    """
    /run 입력 예시:
    {
      "input": {
        "image_url": "https://.../input.png",   # 또는 "image_b64": "..."
        "frames": 16,
        "fps": 8,
        "seed": 123,
        "guidance": 1.0,
        "output_presigned_url": "https://S3-PUT-URL"  # 선택
      }
    }
    """
    inp = job.get("input", {})
    img_ref = inp.get("image_url") or inp.get("image_b64")
    if not img_ref:
        return {"error": "image_url or image_b64 is required"}

    frames = int(inp.get("frames", 16))
    fps = int(inp.get("fps", 8))
    seed = inp.get("seed")
    guidance = float(inp.get("guidance", 1.0))

    in_path = _download_to_tmp(img_ref)
    out_path = str(OUT_DIR / (os.path.basename(in_path) + ".mp4"))

    ok = run_inference_animatediff(
        in_path,
        out_path,
        num_frames=frames,
        fps=fps,
        guidance_scale=guidance,
        seed=seed,
    )
    if not ok:
        return {"error": "inference_failed"}

    # presigned URL 업로드 옵션
    put_url = inp.get("output_presigned_url")
    if put_url:
        with open(out_path, "rb") as f:
            requests.put(put_url, data=f, headers={"Content-Type": "video/mp4"}, timeout=120)
        return {"video_url": put_url}

    # 임시: 파일 경로 반환(워커 종료 시 사라질 수 있음)
    return {"video_path": out_path}


if __name__ == "__main__":
    # Serverless 런타임 진입
    runpod.serverless.start({"handler": handler})

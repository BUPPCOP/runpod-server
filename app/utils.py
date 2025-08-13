# app/utils.py
from typing import List
from PIL import Image
import imageio.v3 as iio

def save_mp4(frames: List[Image.Image], out_path: str, fps: int = 8):
    """PIL 이미지 리스트를 mp4로 저장"""
    # RGB numpy 배열로 변환
    arrs = [ (f.convert("RGB")) for f in frames ]
    # imageio.v3는 바로 PIL 이미지를 받아도 내부에서 ndarray로 변환
    iio.imwrite(out_path, arrs, fps=fps, codec="libx264", format="FFMPEG")
    return out_path

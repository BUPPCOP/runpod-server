from typing import List
from PIL import Image
import numpy as np
import imageio.v3 as iio

def save_mp4(frames: List[Image.Image], out_path: str, fps: int = 8):
    """PIL 이미지 리스트를 mp4로 저장"""
    arrs = [np.asarray(i.convert("RGB")) for i in frames]  # ★ PIL→NumPy
    iio.imwrite(out_path, arrs, fps=fps, codec="libx264", format="FFMPEG")
    return out_path

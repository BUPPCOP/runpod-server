# app/utils.py
import os


def save_mp4(frames, path: str, fps: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import imageio

    # imageio-ffmpeg 필요
    imageio.mimwrite(path, frames, fps=fps, quality=8)

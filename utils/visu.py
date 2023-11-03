from os.path import join as pjoin
from pathlib import Path
from PIL import Image
import os
import numpy as np

def save_imgs_to_video(output_path="video", video_name=""):
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{output_path}/*.png' -c:v libx264 -pix_fmt yuv420p {video_name}.mp4 > /dev/null 2>&1")
    # os.system(f"rm -r {output_path}")

class SimpleVideoRecorder():
    def __init__(self, camera, output="output.mp4"):
        self.camera = camera
        self.counter = 0
        tmp = Path(output.parent) / "tmp" 
        if not tmp.exists():
            # tmp.mkdir()
            tmp.mkdir(parents=True, exist_ok=True)
        self.folder = tmp
        self.output_fname = output

    def render(self):
        self.camera.take_picture()
        color = self.camera.get_color_rgba()
        fname = self.folder / f"output-{self.counter:04d}.png"
        Image.fromarray((color[..., :3].clip(0, 1) * 255).astype(np.uint8)).save(fname)
        self.counter = self.counter + 1

    def dump(self):
        fpath = self.folder.absolute()
        os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{fpath}/output-*.png' -c:v libx264 -pix_fmt yuv420p {self.output_fname}.mp4 > /dev/null 2>&1")
        os.system(f"rm -r {fpath}")
"""Create explicitly synthetic RGB media for platform engineering checks."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
from PIL import Image
import cv2

def create_media(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / 'synthetic_rgb.png'
    video_path = destination / 'synthetic_rgb_50fps.avi'
    if image_path.exists() or video_path.exists():
        raise FileExistsError('Example media already exists; select a new directory')
    width, height, fps, frames = 480, 270, 50., 6
    x = np.linspace(0., 1., width)[None, :]
    y = np.linspace(0., 1., height)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (220 * x + 20).astype(np.uint8)
    rgb[..., 1] = (190 * y + 20).astype(np.uint8)
    rgb[..., 2] = (160 * (1 - x) + 20).astype(np.uint8)
    rgb[10:55, 10:65] = (255, 0, 0)
    rgb[10:55, 75:130] = (0, 255, 0)
    rgb[10:55, 140:195] = (0, 0, 255)
    Image.fromarray(rgb).save(image_path)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MJPG writer unavailable')
    try:
        for index in range(frames):
            frame = rgb.copy()
            col = 230 + index * 24
            frame[85:220, col:col+30] = (250, 250, 250)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    metadata = {'input_class': 'synthetic_engineering_media', 'animal_data': False,
                'generator': 'numpy gradient, explicit RGB swatches and moving stripe; OpenCV MJPG',
                'size_pixels': [width, height], 'video_fps': fps, 'video_frame_count': frames,
                'video_duration_seconds': frames / fps,
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (image_path, video_path)}}
    (destination / 'media_sources.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'examples/platform_media')
    args = parser.parse_args()
    print(json.dumps(create_media(args.output), indent=2))

"""Targeted platform acceptance; run --run-core only for an authorized short integration."""
from pathlib import Path
import argparse
import csv
import json
import queue
import tempfile
import time
import numpy as np
from PIL import Image
from .config import RunConfig
from .media import MediaSource
from .results import ResultStore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-core', action='store_true')
    args = parser.parse_args()
    import cv2
    with tempfile.TemporaryDirectory(prefix='retina_platform_checks_', dir=Path(__file__).resolve().parents[1]) as directory:
        directory = Path(directory)
        image = directory / 'synthetic_red.png'
        pixels = np.zeros((48, 64, 3), np.uint8); pixels[:, :, 0] = 255
        Image.fromarray(pixels).save(image)
        video = directory / 'synthetic_colors.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 50., (64, 48))
        if not writer.isOpened():
            raise RuntimeError('测试视频编码器不可用')
        for channel in (2, 1, 0):
            bgr = np.zeros((48, 64, 3), np.uint8); bgr[:, :, channel] = 255; writer.write(bgr)
        writer.release()
        with MediaSource(image) as source:
            assert source.info['kind'] == 'image' and np.array_equal(source.frame_at(.5), pixels)
            config = RunConfig(media_path=str(image)); config.validate(source)
            assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
        with MediaSource(video) as source:
            assert source.info['fps'] == 50. and source.info['frame_count'] == 3
            for t, rgb_channel in ((0., 0), (.025, 1), (.045, 2)):
                frame = source.frame_at(t)
                assert int(np.argmax(frame.mean(axis=(0, 1)))) == rgb_channel
            RunConfig(media_path=str(video), duration_seconds=.05).validate(source)
            try:
                RunConfig(media_path=str(video), duration_seconds=.2).validate(source)
            except ValueError:
                pass
            else:
                raise AssertionError('Out-of-range video accepted')
        try:
            RunConfig(media_path=str(image), window_seconds=.00015).validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Off-author-grid window accepted')
        print('PASS: synthetic image/video decode, RGB channel, FPS, config roundtrip and invalid inputs')
        if args.run_core:
            from .backend import SimulationSession
            session = SimulationSession(RunConfig(media_path=str(image)), args.project_root).start()
            while session.is_alive():
                try:
                    event = session.events.get(timeout=.1)
                    print(json.dumps(event, ensure_ascii=False))
                    if event['kind'] == 'error':
                        raise AssertionError(event['message'])
                except queue.Empty:
                    pass
            assert session.status == 'completed', session.status
            result = ResultStore(session.path)
            assert len(result.manifest['required_layers']) == 20
            assert len(result.manifest['components']) == 39
            assert len(result.windows) == 3
            window = result.read_window()
            assert not any(k.startswith(('state.', 'state:')) for k in window['signals'])
            assert all(np.all(np.isfinite(v)) for v in window['signals'].values())
            key = next(iter(window['signals']))
            t, v = result.history(key, 0)
            assert len(t) == 7 and np.all(np.diff(t) > 0)
            result.export_curve(key, 0, directory / 'curve.csv')
            result.export_bundle(directory / 'bundle')
            assert (directory / 'bundle' / 'final_checkpoint.json').is_file()
            print('PASS: actual 20 layers / 39 components, continuous windows, curve and checkpoint bundle', session.path)

if __name__ == '__main__':
    main()

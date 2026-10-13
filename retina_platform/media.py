"""Bounded actual media decoding. OpenCV frames are explicitly converted BGR -> RGB."""
from pathlib import Path
import math
import threading
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

class MediaSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError('媒体文件不存在: ' + str(self.path))
        self._lock = threading.RLock()
        self._capture = None
        self._image = None
        self._last_index = None
        self._last_frame = None
        self.last_timestamp_seconds = 0.
        try:
            with Image.open(self.path) as image:
                self._image = np.array(ImageOps.exif_transpose(image).convert('RGB'))
            height, width = self._image.shape[:2]
            self.info = {'kind': 'image', 'width': width, 'height': height, 'fps': None,
                         'frame_count': 1, 'duration_seconds': None, 'path': str(self.path),
                         'color': 'RGB uint8; assumed sRGB', 'timestamp_source': 'static image'}
        except (UnidentifiedImageError, OSError):
            import cv2
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                cap.release()
                raise ValueError('无法读取图像或解码视频')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0 or count <= 0 or width <= 0 or height <= 0:
                cap.release()
                raise ValueError('视频缺少有效 FPS/帧数/尺寸，不能猜测时间轴')
            self._capture = cap
            self.info = {'kind': 'video', 'width': width, 'height': height, 'fps': fps,
                         'frame_count': count, 'duration_seconds': count / fps, 'path': str(self.path),
                         'color': 'decoded BGR explicitly converted to RGB; assumed sRGB',
                         'timestamp_source': 'OpenCV decoded presentation milliseconds; index/FPS fallback recorded',
                         'timeline': 'constant-FPS frame selection; variable-FPS resampling unsupported',
                         'timestamp_fallback_used': False}
            self.frame_at(0.)

    def frame_at(self, seconds):
        if not math.isfinite(float(seconds)) or seconds < 0:
            raise ValueError('媒体时间必须为非负有限秒数')
        with self._lock:
            if self._image is not None:
                return self._image.copy()
            if self._capture is None:
                raise ValueError('媒体已关闭')
            if seconds > self.info['duration_seconds'] + 1e-9:
                raise ValueError('请求时间超过视频范围')
            import cv2
            index = min(int(math.floor(seconds * self.info['fps'] + 1e-8)), self.info['frame_count'] - 1)
            if index != self._last_index:
                if self._last_index is None or index != self._last_index + 1:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = self._capture.read()
                if not ok or bgr is None:
                    raise ValueError('视频帧解码失败: ' + str(index))
                timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
                if not math.isfinite(timestamp) or (index > 0 and timestamp <= 0):
                    timestamp = index / self.info['fps']
                    self.info['timestamp_fallback_used'] = True
                self.last_timestamp_seconds = timestamp
                self._last_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self._last_index = index
            return self._last_frame.copy()

    def close(self):
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

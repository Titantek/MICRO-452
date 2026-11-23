"""Camera access: open a webcam and grab BGR frames, or load test images from disk."""

import os

import cv2


class Camera:
    """Thin wrapper around cv2.VideoCapture.

    Usage:
        with Camera(index=0) as camera:
            frame = camera.read()
    """

    def __init__(self, index=0, width=1920, height=1080, backend=None, warmup_frames=5):
        """
        index         : camera index (0 = default webcam) or a video file path.
        width, height : requested resolution (the camera may pick the closest one it supports).
        backend       : cv2.CAP_* backend (e.g. cv2.CAP_DSHOW or cv2.CAP_MSMF on Windows).
                        None = OpenCV's default. If the given backend fails, the default is tried.
        warmup_frames : frames discarded after opening, while exposure / white balance settle.
        """
        self.index = index
        self.width = width
        self.height = height
        self.backend = cv2.CAP_ANY if backend is None else backend
        self.warmup_frames = warmup_frames
        self.capture = None

    def open(self):
        if self.is_open:
            return self
        backends = [self.backend] if self.backend == cv2.CAP_ANY else [self.backend, cv2.CAP_ANY]
        for backend in backends:
            self.capture = cv2.VideoCapture(self.index, backend)
            if self.capture.isOpened():
                break
            self.capture.release()
            self.capture = None
        if self.capture is None:
            raise RuntimeError(f"Could not open camera {self.index!r} (is it connected / used by another app?)")
        if self.width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(self.warmup_frames):
            self.capture.read()
        return self

    @property
    def is_open(self):
        return self.capture is not None and self.capture.isOpened()

    @property
    def resolution(self):
        """Actual (width, height) delivered by the camera."""
        if not self.is_open:
            return None
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        """Return the next BGR frame. Opens the camera on first use."""
        if not self.is_open:
            self.open()
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the camera")
        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()

    def __del__(self):
        self.release()


def load_image(path):
    """Load a BGR image from disk, raising a clear error instead of returning None."""
    image = cv2.imread(os.fspath(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image {path!r}")
    return image

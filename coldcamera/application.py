"""
Qt-agnostic application controller for ColdCamera.

Owns the processing pipeline, media state, and all business logic.
Can be used by the GUI window or a future CLI runner.
"""

import platform
from typing import List, Optional, Tuple

import numpy as np

from coldcamera.classes.pipeline import ProcessingPipeline
from coldcamera.classes.video_provider import VideoFrameProvider
from coldcamera.config import APPLICATION_VERSION
from coldcamera.core.image_processor import ImageProcessor
from coldcamera.core.media_service import MediaService
from coldcamera.logger import initialize_logger, logger


class Application:
    """
    Core application controller.  Entirely Qt-free.

    Manages the :class:`ProcessingPipeline`, the currently loaded media
    (single image, GIF frame sequence, or video), and exposes methods
    for opening / exporting / processing that both the GUI window and a
    future CLI can call.
    """

    def __init__(self) -> None:
        initialize_logger()
        logger.info("Start application")
        logger.debug(f"Application version: {APPLICATION_VERSION}")
        logger.debug(f"Platform: {platform.system()} {platform.release()} ({platform.architecture()[0]})")

        self.pipeline = ProcessingPipeline()

        # Media state — at most one of these is non-None at a time.
        self.original_image: Optional[np.ndarray] = None
        self.original_frames: Optional[List[np.ndarray]] = None
        self.video_provider: Optional[VideoFrameProvider] = None
        self.frames_fps: int = 10

        self._current_frame_index: int = 0

        logger.success("Application is fully initialized!")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_frame_index(self) -> int:
        """The frame index currently being displayed / processed."""

        return self._current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, value: int) -> None:
        self._current_frame_index = value

    @property
    def has_media(self) -> bool:
        """Whether any media (image, GIF, or video) is currently loaded."""

        return self.original_image is not None or self.original_frames is not None or self.video_provider is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear_media(self) -> None:
        """Release all current media state."""

        if self.video_provider is not None:
            self.video_provider.release()
        self.original_image = None
        self.original_frames = None
        self.video_provider = None
        self._current_frame_index = 0

    # ------------------------------------------------------------------
    # Media loading
    # ------------------------------------------------------------------
    def open_image(self, path: str) -> np.ndarray:
        """
        Load a single image file.

        :param path: File path (png, jpg, bmp …).
        :return: RGBA numpy array of the loaded image.
        """

        self._clear_media()
        arr = MediaService.load_image(path)
        self.original_image = arr
        logger.info(f"Image opened: {path}")
        return arr

    def open_gif(self, path: str) -> Tuple[List[np.ndarray], int]:
        """
        Load a GIF and extract all frames.

        :param path: Path to the GIF file.
        :return: ``(frames, fps)`` — list of RGBA numpy frames and playback FPS.
        """

        self._clear_media()
        frames, fps = MediaService.load_gif_frames(path)
        self.original_frames = frames
        self.frames_fps = fps
        logger.info(f"GIF opened: {path} ({len(frames)} frames, {fps} fps)")
        return frames, fps

    def open_video(self, path: str) -> Tuple[int, int]:
        """
        Open a video file.

        :param path: Path to the video file.
        :return: ``(frame_count, fps)`` of the opened video.
        """

        self._clear_media()
        self.video_provider = MediaService.load_video(path)
        self.frames_fps = int(self.video_provider.fps)
        logger.info(f"Video opened: {path} ({self.video_provider.frame_count} frames, {self.frames_fps} fps)")
        return self.video_provider.frame_count, self.frames_fps

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------
    def get_original_frame(self, frame_index: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Return the original (unprocessed) frame for *frame_index*.

        For a single image the index is ignored.
        For GIF / video, falls back to :pyattr:`current_frame_index`
        when *frame_index* is ``None``.

        :param frame_index: Optional explicit frame index.
        :return: RGBA numpy array, or ``None`` if nothing is loaded.
        """

        if self.original_image is not None:
            return self.original_image

        idx = frame_index if frame_index is not None else self._current_frame_index

        if self.video_provider is not None:
            if 0 <= idx < self.video_provider.frame_count:
                return self.video_provider.get_frame(idx)
            return None

        if self.original_frames is not None:
            if 0 <= idx < len(self.original_frames):
                return self.original_frames[idx]
            return None

        return None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_current(self, frame_index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process the current (or specified) frame through the pipeline.

        :param frame_index: If given, overrides :pyattr:`current_frame_index`.
        :return: ``(original_frame, processed_frame)`` — both as RGBA
                 numpy arrays, or ``(None, None)`` if no media is loaded.
        """

        if frame_index is not None:
            self._current_frame_index = frame_index

        original = self.get_original_frame(self._current_frame_index)
        if original is None:
            return None, None

        processed = ImageProcessor.process_frame(self.pipeline, original)
        if processed is not None:
            processed = ImageProcessor.ensure_rgba(processed)

        return original, processed

    # ------------------------------------------------------------------
    # Exporting
    # ------------------------------------------------------------------
    def export_image(self, path: str) -> None:
        """
        Export the currently loaded image (processed) to *path*.

        :param path: Destination file path.
        :raises ValueError: If no image is loaded or processing fails.
        """

        if self.original_image is None:
            raise ValueError("No image loaded to export")

        processed = ImageProcessor.process_frame(self.pipeline, self.original_image)
        if processed is None:
            raise ValueError("Processing returned no result")

        MediaService.export_image(processed, path)
        logger.info(f"Image exported: {path}")

    def export_gif(self, path: str) -> None:
        """
        Export the currently loaded GIF (processed) to *path*.

        :param path: Destination file path.
        :raises ValueError: If no GIF is loaded.
        """

        if not self.original_frames:
            raise ValueError("No GIF loaded to export")

        MediaService.export_gif(
            self.original_frames,
            self.pipeline,
            self.frames_fps,
            path,
        )
        logger.info(f"GIF exported: {path}")

    def export_video(self, path: str) -> None:
        """
        Export the currently loaded video (processed) to *path*.

        :param path: Destination file path.
        :raises ValueError: If no video is loaded.
        """

        if not self.video_provider:
            raise ValueError("No video loaded to export")

        MediaService.export_video(self.video_provider, self.pipeline, path)
        logger.info(f"Video exported: {path}")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def save_preset(self, path: str) -> None:
        """
        Save the current pipeline as a JSON preset.

        :param path: Destination file path.
        """

        self.pipeline.save_preset(path)
        logger.info(f"Preset saved: {path}")

    def load_preset(self, path: str) -> ProcessingPipeline:
        """
        Load a pipeline preset from a JSON file.

        Replaces :pyattr:`pipeline` with the deserialized pipeline and
        returns it so the caller (e.g. the window) can rebuild the UI.

        :param path: Path to the JSON preset file.
        :return: The newly loaded :class:`ProcessingPipeline`.
        """

        pipeline = ProcessingPipeline.load_preset(path)
        self.pipeline = pipeline
        logger.info(f"Preset loaded: {path}")
        return pipeline


# ======================================================================
# Entry-points
# ======================================================================


def run_gui() -> None:
    """Launch the application in GUI mode (PySide6 / Qt)."""

    import sys

    import qdarktheme
    from PySide6.QtWidgets import QApplication

    from coldcamera.window import MainWindow

    app = Application()
    qt_app = QApplication(sys.argv)

    qdarktheme.setup_theme(
        custom_colors={
            "background": "#191a1c",
            "primary": "#ffffff",
            "border": "#2a2b2b",
        }
    )

    window = MainWindow(app)
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    run_gui()

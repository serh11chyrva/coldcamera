"""
Worker threads for non-blocking operations in ColdCamera.

This module provides QThread-based workers for CPU-intensive tasks:
- Frame processing for viewport updates
- Video/GIF export operations
- Batch processing operations

All workers communicate via Qt signals for thread-safe operations.
"""

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from coldcamera.classes.pipeline import ProcessingPipeline
from coldcamera.classes.video_provider import VideoFrameProvider
from coldcamera.core.image_processor import ImageProcessor
from coldcamera.logger import logger

# ======================================================================
# Frame Processing Worker
# ======================================================================


class FrameProcessWorker(QObject):
    """
    Worker for processing individual frames through the pipeline.

    Runs in a separate thread to avoid blocking the UI during processing.

    Signals:
        finished: Emitted when processing completes with (original, processed) frames.
        error: Emitted if processing fails with error message.
    """

    finished = Signal(np.ndarray, np.ndarray, int)  # original, processed, frame_index
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.pipeline: Optional[ProcessingPipeline] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_index: int = 0
        self._should_stop = False

    def set_data(self, pipeline: ProcessingPipeline, frame: np.ndarray, frame_index: int) -> None:
        """
        Set the data for processing.

        :param pipeline: The processing pipeline to apply.
        :param frame: The original frame to process.
        :param frame_index: Index of the frame being processed.
        """
        self.pipeline = pipeline
        self.frame = frame
        self.frame_index = frame_index
        self._should_stop = False

    def process(self) -> None:
        """Process the frame through the pipeline."""
        try:
            if self._should_stop or self.frame is None or self.pipeline is None:
                return

            original = self.frame.copy()
            processed = ImageProcessor.process_frame(self.pipeline, original)

            if processed is not None and not self._should_stop:
                processed = ImageProcessor.ensure_rgba(processed)
                self.finished.emit(original, processed, self.frame_index)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.error.emit(str(e))

    def stop(self) -> None:
        """Request the worker to stop processing."""
        self._should_stop = True


# ======================================================================
# GIF Export Worker
# ======================================================================


class GifExportWorker(QObject):
    """
    Worker for exporting GIF files with progress reporting.

    Signals:
        progress: Emitted with (current, total) frame counts.
        finished: Emitted when export completes successfully.
        error: Emitted if export fails with error message.
    """

    progress = Signal(int, int)  # current, total
    finished = Signal(str)  # output path
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.original_frames: Optional[List[np.ndarray]] = None
        self.pipeline: Optional[ProcessingPipeline] = None
        self.fps: int = 10
        self.output_path: str = ""
        self._should_stop = False

    def set_data(
        self,
        original_frames: List[np.ndarray],
        pipeline: ProcessingPipeline,
        fps: int,
        output_path: str,
    ) -> None:
        """
        Set the data for export.

        :param original_frames: List of original frames to process.
        :param pipeline: The processing pipeline to apply.
        :param fps: Frames per second for the output GIF.
        :param output_path: Destination file path.
        """
        self.original_frames = original_frames
        self.pipeline = pipeline
        self.fps = fps
        self.output_path = output_path
        self._should_stop = False

    def export(self) -> None:
        """Export the GIF with progress reporting."""
        try:
            if self.original_frames is None or self.pipeline is None:
                self.error.emit("No data set for export")
                return

            from PIL import Image

            processed_frames: List[Image.Image] = []
            total_frames = len(self.original_frames)

            for idx, arr in enumerate(self.original_frames):
                if self._should_stop:
                    self.error.emit("Export cancelled")
                    return

                processed = ImageProcessor.process_frame(self.pipeline, arr)
                if processed is None:
                    continue

                if processed.shape[2] == 4:
                    pil_img = Image.fromarray(processed, "RGBA")
                elif processed.shape[2] == 3:
                    pil_img = Image.fromarray(processed, "RGB")
                else:
                    raise ValueError(f"Unsupported channel count: {processed.shape[2]}")

                processed_frames.append(pil_img)
                self.progress.emit(idx + 1, total_frames)

            if processed_frames and not self._should_stop:
                processed_frames[0].save(
                    self.output_path,
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=int(1000 / self.fps),
                    loop=0,
                    optimize=False,
                )
                self.finished.emit(self.output_path)
            elif not processed_frames:
                self.error.emit("No frames were processed")
        except Exception as e:
            logger.error(f"GIF export error: {e}")
            self.error.emit(str(e))

    def stop(self) -> None:
        """Request the worker to stop exporting."""
        self._should_stop = True


# ======================================================================
# Video Export Worker
# ======================================================================


class VideoExportWorker(QObject):
    """
    Worker for exporting video files with progress reporting.

    Signals:
        progress: Emitted with (current, total) frame counts.
        finished: Emitted when export completes successfully.
        error: Emitted if export fails with error message.
    """

    progress = Signal(int, int)  # current, total
    finished = Signal(str)  # output path
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.video_provider: Optional[VideoFrameProvider] = None
        self.pipeline: Optional[ProcessingPipeline] = None
        self.output_path: str = ""
        self._should_stop = False

    def set_data(
        self,
        video_provider: VideoFrameProvider,
        pipeline: ProcessingPipeline,
        output_path: str,
    ) -> None:
        """
        Set the data for export.

        :param video_provider: Video provider for the source video.
        :param pipeline: The processing pipeline to apply.
        :param output_path: Destination file path.
        """
        self.video_provider = video_provider
        self.pipeline = pipeline
        self.output_path = output_path
        self._should_stop = False

    def export(self) -> None:
        """Export the video with progress reporting."""
        try:
            if self.video_provider is None or self.pipeline is None:
                self.error.emit("No data set for export")
                return

            import cv2

            cap = self.video_provider.cap
            frame_count = self.video_provider.frame_count
            fps = self.video_provider.fps

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                self.error.emit("Could not read video frames")
                return

            h, w, _ = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*"XVID") if self.output_path.lower().endswith(".avi") else cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

            for idx in range(frame_count):
                if self._should_stop:
                    out.release()
                    self.error.emit("Export cancelled")
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = ImageProcessor.process_frame(self.pipeline, frame_rgb)

                if processed is None:
                    continue

                if processed.shape[2] == 4:
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)
                else:
                    processed_rgb = processed

                processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                out.write(processed_bgr)

                self.progress.emit(idx + 1, frame_count)

            out.release()

            if not self._should_stop:
                self.finished.emit(self.output_path)
        except Exception as e:
            logger.error(f"Video export error: {e}")
            self.error.emit(str(e))

    def stop(self) -> None:
        """Request the worker to stop exporting."""
        self._should_stop = True


# ======================================================================
# Thread Manager
# ======================================================================


class WorkerThread(QThread):
    """
    Generic thread wrapper for workers.

    Automatically moves the worker to the thread and cleans up on finish.
    """

    def __init__(self, worker: QObject, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.worker = worker
        self.worker.moveToThread(self)
        self.finished.connect(self.cleanup)

    def cleanup(self) -> None:
        """Clean up the thread and worker."""
        self.worker.deleteLater()
        self.quit()
        self.wait()

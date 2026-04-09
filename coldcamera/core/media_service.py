from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence

from coldcamera.classes.pipeline import ProcessingPipeline
from coldcamera.classes.video_provider import VideoFrameProvider
from coldcamera.core.image_processor import ImageProcessor


class MediaService:
    """
    Service for loading and exporting media files (images, GIFs, videos).

    Operates exclusively on NumPy arrays and PIL — no Qt dependency.
    QImage conversion is handled at the display boundary (window layer).
    """

    # -------------------
    # Loading
    # -------------------
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load an image from disk and return it as an RGBA NumPy array.

        :param path: Path to the image file.
        :return: RGBA numpy array (H, W, 4), dtype uint8.
        """

        pil_image = Image.open(path).convert("RGBA")
        return np.array(pil_image, dtype=np.uint8)

    @staticmethod
    def load_gif_frames(path: str) -> Tuple[List[np.ndarray], int]:
        """
        Load a GIF and extract all frames as RGBA NumPy arrays.

        :param path: Path to the GIF file.
        :return: Tuple of (list of RGBA numpy frames, fps).
        """

        pil_img = Image.open(path)
        frames: List[np.ndarray] = []

        for frame in ImageSequence.Iterator(pil_img):
            rgba = frame.convert("RGBA")
            frames.append(np.array(rgba, dtype=np.uint8))

        duration = pil_img.info.get("duration", 100)
        fps = max(1, int(1000 / duration))

        return frames, fps

    @staticmethod
    def load_video(path: str) -> VideoFrameProvider:
        """
        Open a video file and return a VideoFrameProvider.

        :param path: Path to the video file.
        :return: VideoFrameProvider instance.
        :raises VideoOpenError: If video cannot be opened.
        """

        return VideoFrameProvider(path)

    # -------------------
    # Exporting
    # -------------------
    @staticmethod
    def export_image(frame: np.ndarray, path: str) -> None:
        """
        Export a NumPy frame to disk as an image file.

        :param frame: Processed RGBA or RGB NumPy array (uint8).
        :param path: Destination file path.
        """

        frame = ImageProcessor.ensure_uint8(frame)

        if frame.ndim == 3 and frame.shape[2] == 4:
            pil_img = Image.fromarray(frame, "RGBA").convert("RGB")
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pil_img = Image.fromarray(frame, "RGB")
        else:
            raise ValueError(f"Unsupported array shape for export: {frame.shape}")

        pil_img.save(path)

    @staticmethod
    def export_gif(
        original_frames: List[np.ndarray],
        pipeline: ProcessingPipeline,
        fps: int,
        path: str,
    ) -> None:
        """
        Process and export GIF frames through the pipeline.

        :param original_frames: List of original RGBA NumPy frames.
        :param pipeline: ProcessingPipeline to apply to each frame.
        :param fps: Frames per second for the output GIF.
        :param path: Destination file path.
        :raises ValueError: If a frame has unsupported channel count.
        """

        processed_frames: List[Image.Image] = []

        for arr in original_frames:
            processed = ImageProcessor.process_frame(pipeline, arr)
            if processed is None:
                continue

            if processed.shape[2] == 4:
                pil_img = Image.fromarray(processed, "RGBA")
            elif processed.shape[2] == 3:
                pil_img = Image.fromarray(processed, "RGB")
            else:
                raise ValueError(f"Unsupported channel count: {processed.shape[2]}")
            processed_frames.append(pil_img)

        if processed_frames:
            processed_frames[0].save(
                path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=False,
            )

    @staticmethod
    def export_video(
        video_provider: VideoFrameProvider,
        pipeline: ProcessingPipeline,
        path: str,
    ) -> None:
        """
        Process and export video frames through the pipeline.

        :param video_provider: VideoFrameProvider for the source video.
        :param pipeline: ProcessingPipeline to apply to each frame.
        :param path: Destination file path.
        """

        cap = video_provider.cap
        frame_count = video_provider.frame_count
        fps = video_provider.fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return
        h, w, _ = frame.shape

        fourcc = (
            cv2.VideoWriter_fourcc(*"XVID")  # pyright: ignore[reportAttributeAccessIssue]
            if path.lower().endswith(".avi")
            else cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
        )
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = ImageProcessor.process_frame(pipeline, frame_rgb)

            if processed is None:
                continue

            if processed.shape[2] == 4:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)
            else:
                processed_rgb = processed

            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            out.write(processed_bgr)

        out.release()

from typing import Optional

import cv2
import numpy as np

from coldcamera.classes.pipeline import ProcessingPipeline
from coldcamera.core.operations import CancellationToken, PreviewTile, PreviewTileCallback
from coldcamera.core.processing_settings import ProcessingBackend


class ImageProcessor:
    """
    Service for processing image frames through the pipeline.

    Operates exclusively on NumPy arrays — no Qt dependency.
    QImage conversion is handled at the display boundary (window layer).
    """

    @staticmethod
    def ensure_uint8(arr: np.ndarray) -> np.ndarray:
        """
        Ensure the array is uint8, clipping values if necessary.

        :param arr: Input NumPy array.
        :return: Array with dtype uint8.
        """

        if arr.dtype != np.uint8:
            return np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def ensure_rgba(arr: np.ndarray) -> np.ndarray:
        """
        Ensure a frame has 4 channels (RGBA). Converts 3-channel RGB if needed.

        :param arr: Input NumPy frame.
        :return: RGBA NumPy frame.
        """

        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
        return arr

    @classmethod
    def process_frame(
        cls,
        pipeline: ProcessingPipeline,
        frame: np.ndarray,
        *,
        cache_namespace: object | None = None,
        frame_index: int = 0,
        backend: ProcessingBackend = ProcessingBackend.AUTO,
        cancellation: CancellationToken | None = None,
        tile_callback: PreviewTileCallback | None = None,
        first_dirty_index: int = 0,
        tile_size: int = 256,
    ) -> Optional[np.ndarray]:
        """
        Apply the pipeline to a single NumPy frame.

        :param pipeline: ProcessingPipeline instance.
        :param frame: Input image as NumPy array (typically RGBA uint8).
        :return: Processed NumPy array, or None if input is None.
        """

        if frame is None:
            return None

        def rgba_tile(tile: PreviewTile) -> None:
            if tile_callback is None:
                return
            pixels = cls.ensure_rgba(cls.ensure_uint8(tile.pixels))
            tile_callback(PreviewTile(tile.x, tile.y, tile.full_width, tile.full_height, np.ascontiguousarray(pixels)))

        if len(pipeline.effects) == 0:
            result = frame.copy()
            if tile_callback is not None:
                rgba_tile(PreviewTile(0, 0, result.shape[1], result.shape[0], result))
            return result

        result = pipeline.apply_once(
            frame,
            cache_namespace=cache_namespace,
            frame_index=frame_index,
            backend=backend,
            cancellation=cancellation,
            tile_callback=rgba_tile if tile_callback is not None else None,
            first_dirty_index=first_dirty_index,
            tile_size=tile_size,
        )
        return cls.ensure_uint8(result)

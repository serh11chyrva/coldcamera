"""Qt-independent operation cancellation and progress contracts."""

from __future__ import annotations

from threading import Event
from dataclasses import dataclass
from typing import Callable

import numpy as np


class OperationCancelled(Exception):
    """Raised when a cancellable backend operation is cancelled."""


class CancellationToken:
    """Thread-safe cancellation token shared by an operation and its caller."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled("Operation cancelled")


ProgressCallback = Callable[[int, int], None]
PreviewTileCallback = Callable[["PreviewTile"], None]


@dataclass(frozen=True)
class PreviewTile:
    """A completed preview region and the dimensions of its destination frame."""

    x: int
    y: int
    full_width: int
    full_height: int
    pixels: np.ndarray


def report_progress(callback: ProgressCallback | None, current: int, total: int) -> None:
    if callback is not None:
        callback(current, total)

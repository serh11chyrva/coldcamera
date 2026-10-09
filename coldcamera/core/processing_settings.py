"""Processing preferences shared by the Qt shell and media backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProcessingBackend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class ProcessingSettings:
    backend: ProcessingBackend = ProcessingBackend.AUTO
    cache_budget_bytes: int = 512 * 1024 * 1024
    tile_size: int = 256

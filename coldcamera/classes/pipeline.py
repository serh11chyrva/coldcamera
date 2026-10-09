import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np

from coldcamera.classes.effect import EffectBase
from coldcamera.core.operations import CancellationToken, PreviewTile, PreviewTileCallback
from coldcamera.core.processing_settings import ProcessingBackend
from coldcamera.effects.register import get_by_name
from coldcamera.types import ImageSequence, Processable


@dataclass(frozen=True)
class PipelineChange:
    """Describes the earliest effect whose output may have changed."""

    first_dirty_index: int = 0
    reason: str = "changed"


@dataclass(frozen=True)
class PipelineStageMetric:
    effect_name: str
    elapsed_ms: float
    output_bytes: int
    estimated_working_set_bytes: int = 0


class IntermediateFrameCache:
    """Thread-safe byte-bounded LRU for immutable pipeline stage results."""

    def __init__(self, budget_bytes: int = 512 * 1024 * 1024) -> None:
        self._budget_bytes = max(0, int(budget_bytes))
        self._used_bytes = 0
        self._entries: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def budget_bytes(self) -> int:
        with self._lock:
            return self._budget_bytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            self._budget_bytes = max(0, int(budget_bytes))
            self._evict_to_budget()

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        with self._lock:
            value = self._entries.pop(key, None)
            if value is None:
                return None
            self._entries[key] = value
            return value.copy()

    def contains(self, key: tuple[Any, ...]) -> bool:
        """Test an LRU entry without copying a full cached image."""

        with self._lock:
            value = self._entries.pop(key, None)
            if value is None:
                return False
            self._entries[key] = value
            return True

    def put(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) or value.nbytes > self._budget_bytes:
            return
        stored = np.ascontiguousarray(value).copy()
        stored.setflags(write=False)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._used_bytes -= previous.nbytes
            self._entries[key] = stored
            self._used_bytes += stored.nbytes
            self._evict_to_budget()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0

    def _evict_to_budget(self) -> None:
        while self._used_bytes > self._budget_bytes and self._entries:
            _key, value = self._entries.popitem(last=False)
            self._used_bytes -= value.nbytes


INTERMEDIATE_FRAME_CACHE = IntermediateFrameCache()
AUTO_GPU_MIN_PIXELS = 512 * 512


class ProcessingPipeline:
    """
    Manages a sequence (pipeline) of image processing effects.
    Supports both one-off processing (`apply_once`) and streaming (`apply_stream`).
    Includes caching for static inputs.

    :param effects: Optional initial list of EffectBase objects.
    """

    def __init__(self, effects: Optional[List[EffectBase]] = None):
        self.effects: List[EffectBase] = effects or []
        self._cache_input: Optional[Processable] = None
        self._cache_output: Optional[Processable] = None
        self._cache_fingerprint: tuple[Any, ...] | None = None
        self.last_metrics: list[PipelineStageMetric] = []
        self.last_gpu_fallback_effects: list[str] = []
        self.estimated_peak_working_bytes = 0

    def _run_pipeline(self, frame: Processable) -> Processable:
        """
        Apply all enabled effects to a single frame.

        :param frame: Input frame.
        :return: Processed frame.
        """

        output = frame

        for eff in self.effects:
            if getattr(eff, "enabled", True):
                output = eff.apply(output)

        return output

    @staticmethod
    def _stage_signature(effect: EffectBase) -> str:
        payload = effect.to_dictionary()
        payload["enabled"] = bool(effect.enabled)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _stable_seed(cache_namespace: Any, frame_index: int, index: int, effect: EffectBase) -> int:
        payload = f"{cache_namespace!r}|{frame_index}|{index}|{type(effect).__module__}.{type(effect).__qualname__}|{effect.name}"
        return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "little")

    @staticmethod
    def _emit_full_frame(callback: PreviewTileCallback | None, frame: np.ndarray) -> None:
        if callback is None or frame.ndim < 2:
            return
        callback(PreviewTile(0, 0, frame.shape[1], frame.shape[0], np.ascontiguousarray(frame)))

    @staticmethod
    def _apply_chunked(
        effect: EffectBase,
        input_data: np.ndarray,
        *,
        backend: ProcessingBackend,
        tile_size: int,
        cancellation: CancellationToken | None,
        tile_callback: PreviewTileCallback | None,
        is_final: bool,
    ) -> np.ndarray:
        capabilities = effect.get_execution_capabilities()
        use_gpu_full_frame = backend != ProcessingBackend.CPU and effect.supports_gpu()
        if not capabilities.tileable or use_gpu_full_frame or input_data.ndim != 3 or input_data.shape[0] == 0 or input_data.shape[1] == 0:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            result = np.asarray(effect.apply_with_backend(input_data.copy(), backend))
            if is_final and tile_callback is not None:
                ProcessingPipeline._emit_full_frame(tile_callback, result)
            return result

        height, width = input_data.shape[:2]
        tile_size = max(32, int(tile_size))
        halo_x = max(0, int(capabilities.halo_x))
        halo_y = max(0, int(capabilities.halo_y))
        output: np.ndarray | None = None
        output_channels: int | None = None

        for y in range(0, height, tile_size):
            y_end = min(height, y + tile_size)
            for x in range(0, width, tile_size):
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                x_end = min(width, x + tile_size)
                source_x0, source_x1 = max(0, x - halo_x), min(width, x_end + halo_x)
                source_y0, source_y1 = max(0, y - halo_y), min(height, y_end + halo_y)
                patch = np.asarray(effect.apply_with_backend(input_data[source_y0:source_y1, source_x0:source_x1].copy(), backend))
                if patch.ndim != 3 or patch.shape[:2] != (source_y1 - source_y0, source_x1 - source_x0):
                    # A misdeclared effect is safer on the established whole-frame path.
                    if cancellation is not None:
                        cancellation.raise_if_cancelled()
                    result = np.asarray(effect.apply_with_backend(input_data.copy(), backend))
                    if is_final and tile_callback is not None:
                        ProcessingPipeline._emit_full_frame(tile_callback, result)
                    return result

                if output is None:
                    output_channels = patch.shape[2]
                    output = np.empty((height, width, output_channels), dtype=patch.dtype)
                elif patch.shape[2] != output_channels or patch.dtype != output.dtype:
                    if cancellation is not None:
                        cancellation.raise_if_cancelled()
                    result = np.asarray(effect.apply_with_backend(input_data.copy(), backend))
                    if is_final and tile_callback is not None:
                        ProcessingPipeline._emit_full_frame(tile_callback, result)
                    return result

                core_y0, core_y1 = y - source_y0, y_end - source_y0
                core_x0, core_x1 = x - source_x0, x_end - source_x0
                output[y:y_end, x:x_end] = patch[core_y0:core_y1, core_x0:core_x1]

            if is_final and tile_callback is not None and output is not None:
                tile_callback(PreviewTile(0, y, width, height, np.ascontiguousarray(output[y:y_end])))

        return output if output is not None else input_data.copy()

    @staticmethod
    def _estimate_stage_working_bytes(
        input_data: np.ndarray,
        output: np.ndarray,
        effect: EffectBase,
        *,
        backend: ProcessingBackend,
        tile_size: int,
    ) -> int:
        """Estimate resident input/output and the largest stage-local buffers."""

        capabilities = effect.get_execution_capabilities()
        if backend == ProcessingBackend.CPU and capabilities.tileable and input_data.ndim == 3:
            tile_size = max(32, int(tile_size))
            halo_x = max(0, int(capabilities.halo_x))
            halo_y = max(0, int(capabilities.halo_y))
            patch_width = min(input_data.shape[1], tile_size + 2 * halo_x)
            patch_height = min(input_data.shape[0], tile_size + 2 * halo_y)
            patch_bytes = patch_width * patch_height * input_data.shape[2] * input_data.dtype.itemsize
            temporary_bytes = 2 * patch_bytes
        elif backend != ProcessingBackend.CPU and effect.supports_gpu() and input_data.ndim == 3:
            band_height = min(max(32, int(tile_size)), input_data.shape[0])
            band_bytes = input_data.shape[1] * band_height * 4
            # Host staging/output bands plus the pair of RGBA render targets.
            temporary_bytes = 4 * band_bytes
        else:
            # A conservative floor for whole-frame conversions and scratch arrays.
            temporary_bytes = 2 * max(input_data.nbytes, output.nbytes)
        return int(input_data.nbytes + output.nbytes + temporary_bytes)

    @staticmethod
    def _apply_gpu_bands(
        executor: Any,
        effects: list[tuple[int, EffectBase, Any, Any, tuple[Any, ...] | None]],
        input_data: np.ndarray,
        *,
        band_height: int,
        cancellation: CancellationToken | None,
        tile_callback: PreviewTileCallback | None,
        is_final: bool,
    ) -> np.ndarray:
        """Run a fused pointwise GPU segment with bounded framebuffer height."""

        if input_data.ndim != 3 or input_data.shape[2] not in (3, 4):
            return executor.render(input_data, [item[2] for item in effects])
        height, width = input_data.shape[:2]
        band_height = max(32, int(band_height))
        output = np.empty((height, width, input_data.shape[2]), dtype=np.uint8)
        passes = [item[2] for item in effects]
        for y in range(0, height, band_height):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            y_end = min(height, y + band_height)
            band = input_data[y:y_end]
            expected_band_height = min(band_height, height)
            if band.shape[0] < expected_band_height:
                # Keep framebuffer dimensions stable for the final partial band.
                # Pointwise passes do not depend on the padded rows.
                padding = np.repeat(band[-1:], expected_band_height - band.shape[0], axis=0)
                band = np.concatenate((band, padding), axis=0)
            rendered = executor.render(band, passes)
            output[y:y_end] = rendered[: y_end - y]
            if is_final and tile_callback is not None:
                tile_callback(PreviewTile(0, y, width, height, output[y:y_end].copy()))
        return output

    def apply_once(
        self,
        input_data: Processable,
        *,
        cache_namespace: Any | None = None,
        frame_index: int = 0,
        backend: ProcessingBackend = ProcessingBackend.AUTO,
        cancellation: CancellationToken | None = None,
        tile_callback: PreviewTileCallback | None = None,
        first_dirty_index: int = 0,
        tile_size: int = 256,
    ) -> Processable:
        """
        Apply the pipeline to a single static image.
        Uses caching to avoid reprocessing if input is unchanged.

        :param input_data: Input frame.
        :return: Processed frame.
        """

        if not isinstance(backend, ProcessingBackend):
            backend = ProcessingBackend(str(backend).lower())
        pipeline_fingerprint = (backend.value, *(self._stage_signature(effect) + str(bool(effect.enabled)) for effect in self.effects))
        if cache_namespace is None and input_data is self._cache_input and pipeline_fingerprint == self._cache_fingerprint:
            if tile_callback is not None and isinstance(self._cache_output, np.ndarray):
                self._emit_full_frame(tile_callback, self._cache_output)
            return self._cache_output  # pyright: ignore[reportReturnType]

        enabled_effects = [(index, effect) for index, effect in enumerate(self.effects) if getattr(effect, "enabled", True)]
        if not enabled_effects:
            result = input_data.copy() if hasattr(input_data, "copy") else input_data
            if isinstance(result, np.ndarray):
                self._emit_full_frame(tile_callback, result)
            self._cache_input, self._cache_output = input_data, result
            self._cache_fingerprint = pipeline_fingerprint
            return result

        frame = np.asarray(input_data)
        initial_shape = (tuple(frame.shape), str(frame.dtype))
        prefix = hashlib.sha256(repr(initial_shape).encode("utf-8"))
        output: np.ndarray = frame
        self.last_metrics = []
        self.last_gpu_fallback_effects = []
        self.estimated_peak_working_bytes = int(frame.nbytes)
        final_position = enabled_effects[-1][0]

        def make_stage_key(prefix_digest: Any) -> tuple[Any, ...] | None:
            if cache_namespace is None:
                return None
            return (
                cache_namespace,
                int(frame_index),
                backend.value,
                prefix_digest.hexdigest(),
            )

        cursor = 0
        if cache_namespace is not None and first_dirty_index > 0:
            checkpoint_position = -1
            checkpoint_prefix = prefix.copy()
            for position, (original_index, prior_effect) in enumerate(enabled_effects):
                if original_index >= first_dirty_index:
                    break
                checkpoint_prefix.update(self._stage_signature(prior_effect).encode("ascii"))
                checkpoint_position = position
            if checkpoint_position >= 0:
                checkpoint_key = make_stage_key(checkpoint_prefix)
                checkpoint = INTERMEDIATE_FRAME_CACHE.get(checkpoint_key) if checkpoint_key is not None else None
                if checkpoint is not None:
                    output = checkpoint
                    prefix = checkpoint_prefix
                    cursor = checkpoint_position + 1

        while cursor < len(enabled_effects):
            effect_index, effect = enabled_effects[cursor]
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            signature = self._stage_signature(effect)
            stage_prefix = prefix.copy()
            stage_prefix.update(signature.encode("ascii"))
            stage_key = make_stage_key(stage_prefix)

            cached = INTERMEDIATE_FRAME_CACHE.get(stage_key) if stage_key is not None else None
            if cached is not None:
                output = cached
                if effect_index == final_position:
                    self._emit_full_frame(tile_callback, output)
                prefix = stage_prefix
                cursor += 1
                continue

            use_gpu = backend == ProcessingBackend.GPU or (
                backend == ProcessingBackend.AUTO and output.ndim >= 2 and output.shape[0] * output.shape[1] >= AUTO_GPU_MIN_PIXELS
            )
            stage_backend = backend if use_gpu else ProcessingBackend.CPU
            gpu_pass = effect.get_gpu_pass() if use_gpu else None
            if use_gpu and gpu_pass is None and not effect.supports_gpu():
                self.last_gpu_fallback_effects.append(effect.name)
            if gpu_pass is not None:
                from coldcamera.core.gpu import get_gpu_executor

                run_effects: list[tuple[int, EffectBase, Any, Any, tuple[Any, ...] | None]] = [(effect_index, effect, gpu_pass, stage_prefix, stage_key)]
                run_prefix = stage_prefix
                next_cursor = cursor + 1
                while next_cursor < len(enabled_effects):
                    next_index, next_effect = enabled_effects[next_cursor]
                    next_pass = next_effect.get_gpu_pass()
                    if next_pass is None:
                        break
                    next_prefix = run_prefix.copy()
                    next_prefix.update(self._stage_signature(next_effect).encode("ascii"))
                    next_key = make_stage_key(next_prefix)
                    if next_key is not None and INTERMEDIATE_FRAME_CACHE.contains(next_key):
                        break
                    run_effects.append((next_index, next_effect, next_pass, next_prefix, next_key))
                    run_prefix = next_prefix
                    next_cursor += 1

                started = time.perf_counter()
                stage_input = output
                try:
                    output = self._apply_gpu_bands(
                        get_gpu_executor(),
                        run_effects,
                        output,
                        band_height=tile_size,
                        cancellation=cancellation,
                        tile_callback=tile_callback,
                        is_final=run_effects[-1][0] == final_position,
                    )
                    prefix = run_prefix
                    cursor = next_cursor
                    # Keep the completed GPU segment as a reusable cache boundary.
                    segment_key = run_effects[-1][4]
                    if segment_key is not None and output.dtype == np.uint8:
                        INTERMEDIATE_FRAME_CACHE.put(segment_key, output)
                    self.last_metrics.append(
                        PipelineStageMetric(
                            "GPU: " + ", ".join(item[1].name for item in run_effects),
                            (time.perf_counter() - started) * 1000,
                            int(output.nbytes),
                            self._estimate_stage_working_bytes(
                                stage_input,
                                output,
                                run_effects[-1][1],
                                backend=ProcessingBackend.GPU,
                                tile_size=tile_size,
                            ),
                        )
                    )
                    self.estimated_peak_working_bytes = max(self.estimated_peak_working_bytes, self.last_metrics[-1].estimated_working_set_bytes)
                    continue
                except Exception:
                    # Auto and GPU-preferred modes retain image correctness without a usable GL context.
                    for _index, run_effect, _pass, _run_prefix, _run_key in run_effects:
                        if run_effect.name not in self.last_gpu_fallback_effects:
                            self.last_gpu_fallback_effects.append(run_effect.name)
                    output = output.copy()
                    for run_index, run_effect, _pass, run_stage_prefix, _run_key in run_effects:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        if run_effect.get_execution_capabilities().stochastic and cache_namespace is not None:
                            run_effect.set_execution_seed(self._stable_seed(cache_namespace, frame_index, run_index, run_effect))
                        cpu_started = time.perf_counter()
                        stage_input = output
                        try:
                            output = self._apply_chunked(
                                run_effect,
                                output,
                                backend=ProcessingBackend.CPU,
                                tile_size=tile_size,
                                cancellation=cancellation,
                                tile_callback=tile_callback if run_index == final_position else None,
                                is_final=run_index == final_position,
                            )
                        finally:
                            run_effect.set_execution_seed(None)
                        working_set = self._estimate_stage_working_bytes(
                            stage_input,
                            output,
                            run_effect,
                            backend=ProcessingBackend.CPU,
                            tile_size=tile_size,
                        )
                        self.last_metrics.append(
                            PipelineStageMetric(run_effect.name + " (CPU fallback)", (time.perf_counter() - cpu_started) * 1000, int(output.nbytes), working_set)
                        )
                        self.estimated_peak_working_bytes = max(self.estimated_peak_working_bytes, working_set)
                        stage_key_for_cpu = make_stage_key(run_stage_prefix)
                        if stage_key_for_cpu is not None and output.dtype == np.uint8:
                            INTERMEDIATE_FRAME_CACHE.put(stage_key_for_cpu, output)
                    prefix = run_prefix
                    cursor = next_cursor
                    continue

            if effect.get_execution_capabilities().stochastic and cache_namespace is not None:
                effect.set_execution_seed(self._stable_seed(cache_namespace, frame_index, effect_index, effect))
            started = time.perf_counter()
            stage_input = output
            try:
                output = self._apply_chunked(
                    effect,
                    output,
                    backend=stage_backend,
                    tile_size=tile_size,
                    cancellation=cancellation,
                    tile_callback=tile_callback if effect_index == final_position else None,
                    is_final=effect_index == final_position,
                )
            finally:
                effect.set_execution_seed(None)
            elapsed_ms = (time.perf_counter() - started) * 1000
            working_set = self._estimate_stage_working_bytes(stage_input, output, effect, backend=stage_backend, tile_size=tile_size)
            self.last_metrics.append(PipelineStageMetric(effect.name, elapsed_ms, int(output.nbytes), working_set))
            self.estimated_peak_working_bytes = max(self.estimated_peak_working_bytes, working_set)
            if stage_key is not None and (cancellation is None or not cancellation.is_cancelled) and output.dtype == np.uint8:
                INTERMEDIATE_FRAME_CACHE.put(stage_key, output)
            prefix = stage_prefix
            cursor += 1

        result: Processable = output
        if cache_namespace is None:
            self._cache_input, self._cache_output = input_data, result
            self._cache_fingerprint = pipeline_fingerprint
        return result

    def apply_stream(self, input_sequence: Union[Iterator[Processable], ImageSequence.Iterator]) -> Iterator[Processable]:
        """
        Generator: apply pipeline to each frame in a sequence.
        Suitable for GIF, MP4, or real-time sources (camera).

        :param input_sequence: Sequence or iterator of frames.
        :yield: Processed frames one by one.
        """

        for frame in input_sequence:
            if hasattr(frame, "copy"):
                frame = frame.copy()  # pyright: ignore[reportAttributeAccessIssue]
            yield self._run_pipeline(frame)

    def apply_frames(self, frames: List[Processable]) -> List[Processable]:
        """
        Apply pipeline to a list of frames.

        :param frames: List of frames.
        :return: List of processed frames.
        """

        return list(self.apply_stream(frames))  # pyright: ignore[reportArgumentType]

    def add_effect(self, effect: EffectBase) -> None:
        """
        Append effect to the end of the pipeline.

        :param effect: Effect to append to the pipeline.
        """

        self.effects.append(effect)

    @staticmethod
    def find_effect_class(effect_name: str) -> Optional[Type[EffectBase]]:
        """
        Look up an effect class by its display name in the registry.

        :param effect_name: Human-readable effect name (e.g. "Exposure").
        :return: Effect class, or None if not found.
        """

        return get_by_name(effect_name)

    def add_effect_by_name(self, effect_name: str) -> Optional[EffectBase]:
        """
        Instantiate an effect by its display name and append it to the pipeline.

        :param effect_name: Human-readable effect name (e.g. "Exposure").
        :return: The created EffectBase instance, or None if the name was not found.
        """

        effect_cls = self.find_effect_class(effect_name)
        if effect_cls is None:
            return None

        effect = effect_cls()  # pyright: ignore[reportCallIssue]
        self.effects.append(effect)
        return effect

    def reorder_from_effects(self, effects: List[EffectBase]) -> None:
        """
        Replace the current effects list with a new ordered list.

        Useful for syncing the pipeline order after a drag-and-drop reorder in the UI.

        :param effects: New ordered list of EffectBase instances.
        """

        self.effects = effects

    def insert_effect(self, index: int, effect: EffectBase) -> None:
        """
        Insert effect at a specific index in the pipeline.

        :param index: Index where the effect should be inserted.
        :param effect: Effect to insert into the pipeline.
        """

        self.effects.insert(index, effect)

    def remove_effect(self, index: int) -> None:
        """
        Remove effect at a given index.

        :param index: Index of the effect to remove.
        """

        if 0 <= index < len(self.effects):
            del self.effects[index]

    def move_effect(self, old_index: int, new_index: int) -> None:
        """
        Reorder effects by moving one from old_index to new_index.

        :param old_index: Index of the effect to move.
        :param new_index: Index where the effect should be moved to.
        """

        if 0 <= old_index < len(self.effects):
            effect = self.effects.pop(old_index)
            self.effects.insert(new_index, effect)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Serialize the pipeline into a dictionary (preset format).
        Each effect is serialized via its `to_dict` method.

        :return: Dictionary with pipeline description.
        """

        return {"pipeline": [effect.to_dictionary() for effect in self.effects]}

    @classmethod
    def from_dictionary(cls, d: Dict[str, Any]) -> "ProcessingPipeline":
        """
        Deserialize pipeline from dictionary.

        :param d: Dictionary with serialized pipeline data.
        :return: ProcessingPipeline instance.
        :raise ValueError: If an unknown effect type is encountered.
        """

        effects: List[EffectBase] = []

        for eff_data in d.get("pipeline", []):
            eff_name = eff_data.get("type")
            effect_cls = get_by_name(eff_name)

            if effect_cls is None:
                raise ValueError(f"Unknown effect type: {eff_name}")

            effect = effect_cls.from_dictionary(eff_data)
            effects.append(effect)

        return cls(effects)

    def save_preset(self, path: str) -> None:
        """
        Save pipeline as JSON preset to a file.

        :param path: File path to save preset.
        """

        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dictionary(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_preset(cls, path: str) -> "ProcessingPipeline":
        """
        Load pipeline from JSON preset file.

        :param path: Path to JSON preset file.
        :return: ProcessingPipeline instance.
        """

        import json

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dictionary(d)

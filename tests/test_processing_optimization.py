from __future__ import annotations

import unittest
from uuid import uuid4
from unittest.mock import patch

import numpy as np

from coldcamera.classes.effect import EffectBase, EffectCapabilities
from coldcamera.classes.parameter import EffectParam
from coldcamera.classes.pipeline import INTERMEDIATE_FRAME_CACHE, IntermediateFrameCache, ProcessingPipeline
from coldcamera.core.gpu import GPUExecutor, GPU_RENDER_TARGET_BUDGET_BYTES
from coldcamera.effects.includes.blur import BlurEffect
from coldcamera.core.operations import CancellationToken, OperationCancelled, PreviewTile
from coldcamera.core.processing_settings import ProcessingBackend
from coldcamera.effects.includes.contrast_brightness import ContrastBrightnessEffect
from coldcamera.effects.includes.exposure import ExposureEffect
from coldcamera.effects.includes.noise import NoiseEffect
from coldcamera.effects.includes.sharpen import SharpenEffect
from coldcamera.effects.includes.warmth import WarmthEffect


class _CountingEffect(EffectBase):
    calls: dict[str, int] = {}

    def __init__(self, name: str = "count", amount: int = 1):
        super().__init__(name, params=[EffectParam("amount", int, amount)])

    def apply(self, input_data):
        type(self).calls[self.name] = type(self).calls.get(self.name, 0) + 1
        return np.clip(np.asarray(input_data, dtype=np.int16) + self.get_parameter("amount"), 0, 255).astype(np.uint8)

    def get_execution_capabilities(self) -> EffectCapabilities:
        return EffectCapabilities(locality="pointwise")


class ProcessingOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        INTERMEDIATE_FRAME_CACHE.clear()
        INTERMEDIATE_FRAME_CACHE.set_budget(512 * 1024 * 1024)
        _CountingEffect.calls.clear()

    def test_cpu_tiles_match_whole_frame_for_pointwise_and_local_effects(self) -> None:
        image = np.random.default_rng(12).integers(0, 256, (79, 91, 4), dtype=np.uint8)
        effects = [ExposureEffect(), SharpenEffect()]
        effects[0].set_parameter("exposure", 0.73)
        effects[1].set_parameter("amount", 1.7)
        effects[1].set_parameter("radius", 5)

        full = image
        for effect in effects:
            full = effect.apply(full)

        emitted: list[PreviewTile] = []
        tiled = ProcessingPipeline(effects).apply_once(
            image,
            backend=ProcessingBackend.CPU,
            tile_size=32,
            tile_callback=emitted.append,
        )

        np.testing.assert_array_equal(tiled, full)
        self.assertEqual(len(emitted), 3)
        rebuilt = np.zeros_like(tiled)
        for tile in emitted:
            rebuilt[tile.y : tile.y + tile.pixels.shape[0], tile.x : tile.x + tile.pixels.shape[1]] = tile.pixels
        np.testing.assert_array_equal(rebuilt, tiled)

    def test_dirty_index_reuses_unchanged_pipeline_prefix(self) -> None:
        source_key = ("cache-test", uuid4().hex)
        image = np.full((8, 8, 4), 20, dtype=np.uint8)
        first = _CountingEffect("first", 3)
        second = _CountingEffect("second", 4)
        ProcessingPipeline([first, second]).apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key)

        revised_first = _CountingEffect("first", 3)
        revised_second = _CountingEffect("second", 9)
        revised = ProcessingPipeline([revised_first, revised_second])
        result = revised.apply_once(
            image,
            backend=ProcessingBackend.CPU,
            cache_namespace=source_key,
            first_dirty_index=1,
        )

        self.assertEqual(_CountingEffect.calls, {"first": 1, "second": 2})
        self.assertEqual(int(result[0, 0, 0]), 32)

    def test_removing_an_effect_reuses_only_the_unchanged_prefix(self) -> None:
        source_key = ("remove-test", uuid4().hex)
        image = np.full((8, 8, 4), 20, dtype=np.uint8)
        first = _CountingEffect("first", 3)
        removed = _CountingEffect("removed", 4)
        last = _CountingEffect("last", 5)
        ProcessingPipeline([first, removed, last]).apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key)

        revised = ProcessingPipeline([_CountingEffect("first", 3), _CountingEffect("last", 5)])
        revised.apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key, first_dirty_index=1)

        self.assertEqual(_CountingEffect.calls, {"first": 1, "removed": 1, "last": 2})

    def test_inserting_an_effect_preserves_the_prefix_cache(self) -> None:
        source_key = ("insert-test", uuid4().hex)
        image = np.full((8, 8, 4), 20, dtype=np.uint8)
        ProcessingPipeline([_CountingEffect("first", 3), _CountingEffect("last", 5)]).apply_once(
            image, backend=ProcessingBackend.CPU, cache_namespace=source_key
        )

        revised = ProcessingPipeline([_CountingEffect("first", 3), _CountingEffect("inserted", 7), _CountingEffect("last", 5)])
        revised.apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key, first_dirty_index=1)

        self.assertEqual(_CountingEffect.calls, {"first": 1, "last": 2, "inserted": 1})

    def test_reordering_effects_invalidates_the_changed_suffix(self) -> None:
        source_key = ("reorder-test", uuid4().hex)
        image = np.full((8, 8, 4), 20, dtype=np.uint8)
        first = _CountingEffect("first", 3)
        second = _CountingEffect("second", 5)
        ProcessingPipeline([first, second]).apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key)

        reordered = ProcessingPipeline([_CountingEffect("second", 5), _CountingEffect("first", 3)])
        reordered.apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key, first_dirty_index=0)

        self.assertEqual(_CountingEffect.calls, {"first": 2, "second": 2})

    def test_disabling_and_reenabling_an_effect_keeps_earlier_cache(self) -> None:
        source_key = ("toggle-test", uuid4().hex)
        image = np.full((8, 8, 4), 20, dtype=np.uint8)
        ProcessingPipeline([_CountingEffect("first", 3), _CountingEffect("second", 5)]).apply_once(
            image, backend=ProcessingBackend.CPU, cache_namespace=source_key
        )

        first, second = _CountingEffect("first", 3), _CountingEffect("second", 5)
        second.enabled = False
        revised = ProcessingPipeline([first, second])
        without_second = revised.apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key, first_dirty_index=1)
        second.enabled = True
        with_second = revised.apply_once(image, backend=ProcessingBackend.CPU, cache_namespace=source_key, first_dirty_index=1)

        self.assertEqual(int(without_second[0, 0, 0]), 23)
        self.assertEqual(int(with_second[0, 0, 0]), 28)
        self.assertEqual(_CountingEffect.calls, {"first": 1, "second": 1})

    def test_cpu_tiled_blur_matches_full_frame_at_image_edges(self) -> None:
        image = np.random.default_rng(13).integers(0, 256, (83, 91, 4), dtype=np.uint8)
        effect = BlurEffect()
        effect.set_parameter("amount", 4.5)
        effect.set_parameter("angle", 31.0)
        effect.set_parameter("opacity", 0.63)

        expected = effect.apply_with_backend(image, ProcessingBackend.CPU)
        actual = ProcessingPipeline([effect]).apply_once(image, backend=ProcessingBackend.CPU, tile_size=32)

        np.testing.assert_array_equal(actual, expected)

    def test_intermediate_cache_evicts_by_byte_budget(self) -> None:
        cache = IntermediateFrameCache(8)
        cache.put(("first",), np.zeros((2, 2), dtype=np.uint8))
        cache.put(("second",), np.ones((2, 2), dtype=np.uint8))
        self.assertIsNotNone(cache.get(("first",)))
        cache.put(("third",), np.full((2, 2), 2, dtype=np.uint8))

        self.assertLessEqual(cache.used_bytes, cache.budget_bytes)
        self.assertIsNotNone(cache.get(("first",)))
        self.assertIsNone(cache.get(("second",)))
        self.assertIsNotNone(cache.get(("third",)))

    def test_stochastic_effect_is_reproducible_without_cache(self) -> None:
        INTERMEDIATE_FRAME_CACHE.set_budget(0)
        image = np.random.default_rng(91).integers(0, 256, (40, 45, 4), dtype=np.uint8)
        effect = NoiseEffect()
        effect.set_parameter("strength", 22.0)
        pipeline = ProcessingPipeline([effect])

        first = pipeline.apply_once(image, cache_namespace=("seed-test", "image"), backend=ProcessingBackend.CPU)
        second = pipeline.apply_once(image, cache_namespace=("seed-test", "image"), backend=ProcessingBackend.CPU)
        np.testing.assert_array_equal(first, second)

    def test_chunk_cancellation_is_checked_between_bands(self) -> None:
        image = np.full((96, 64, 4), 64, dtype=np.uint8)
        cancellation = CancellationToken()

        def cancel_after_first_band(_tile: PreviewTile) -> None:
            cancellation.cancel()

        with self.assertRaises(OperationCancelled):
            ProcessingPipeline([ExposureEffect()]).apply_once(
                image,
                backend=ProcessingBackend.CPU,
                tile_size=32,
                cancellation=cancellation,
                tile_callback=cancel_after_first_band,
            )

    def test_gpu_pointwise_chain_stays_close_to_cpu_reference(self) -> None:
        for channels, height, width in ((3, 37, 41), (4, 65, 57)):
            with self.subTest(channels=channels, size=(width, height)):
                image = np.random.default_rng(110 + channels).integers(0, 256, (height, width, channels), dtype=np.uint8)
                cpu_effects = [ExposureEffect(), ContrastBrightnessEffect(), WarmthEffect()]
                gpu_effects = [ExposureEffect(), ContrastBrightnessEffect(), WarmthEffect()]
                for effects in (cpu_effects, gpu_effects):
                    effects[0].set_parameter("exposure", 1.27)
                    effects[1].set_parameter("contrast", 1.16)
                    effects[1].set_parameter("brightness", 7.0)
                    effects[2].set_parameter("warmth", 24.0)

                expected = ProcessingPipeline(cpu_effects).apply_once(image, backend=ProcessingBackend.CPU)
                gpu_pipeline = ProcessingPipeline(gpu_effects)
                try:
                    actual = gpu_pipeline.apply_once(image, backend=ProcessingBackend.GPU, tile_size=32)
                except Exception as exc:
                    self.skipTest(f"standalone GL context is unavailable: {exc}")
                if gpu_pipeline.last_gpu_fallback_effects:
                    self.skipTest("GPU effects fell back to CPU on this environment")

                self.assertEqual(actual.shape, expected.shape)
                self.assertLessEqual(int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))), 1)

    def test_gpu_band_padding_keeps_framebuffer_dimensions_stable(self) -> None:
        class RecordingExecutor:
            def __init__(self) -> None:
                self.shapes: list[tuple[int, ...]] = []

            def render(self, image, _passes):
                self.shapes.append(image.shape)
                return image.copy()

        image = np.zeros((600, 23, 3), dtype=np.uint8)
        executor = RecordingExecutor()
        effect = ExposureEffect()
        tiles: list[PreviewTile] = []
        result = ProcessingPipeline._apply_gpu_bands(
            executor,
            [(0, effect, effect.get_gpu_pass(), None, None)],
            image,
            band_height=256,
            cancellation=None,
            tile_callback=tiles.append,
            is_final=True,
        )

        self.assertEqual(executor.shapes, [(256, 23, 3)] * 3)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual([tile.pixels.shape[0] for tile in tiles], [256, 256, 88])

    def test_gpu_target_allocation_obeys_vram_budget(self) -> None:
        self.assertEqual(GPUExecutor._validate_render_target_size(4096, 1024, 16384), 32 * 1024 * 1024)
        with self.assertRaises(MemoryError):
            GPUExecutor._validate_render_target_size(6000, 6000, 16384)
        self.assertGreater(GPU_RENDER_TARGET_BUDGET_BYTES, 0)

    def test_gpu_failure_falls_back_to_cpu(self) -> None:
        class BrokenExecutor:
            def render(self, *_args, **_kwargs):
                raise RuntimeError("no GL context")

        image = np.full((9, 11, 4), 80, dtype=np.uint8)
        pipeline = ProcessingPipeline([ExposureEffect()])
        with patch("coldcamera.core.gpu.get_gpu_executor", return_value=BrokenExecutor()):
            actual = pipeline.apply_once(image, backend=ProcessingBackend.GPU)

        expected = ExposureEffect()
        expected.set_parameter("exposure", 1.0)
        np.testing.assert_array_equal(actual, expected.apply(image))
        self.assertEqual(pipeline.last_gpu_fallback_effects, ["Exposure"])


if __name__ == "__main__":
    unittest.main()

"""Long-lived, thread-affine ModernGL executor for image shader passes."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import moderngl
import numpy as np

from coldcamera.classes.effect import GPUShaderPass


VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 vUv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    vUv = in_uv;
}
"""

GPU_RENDER_TARGET_BUDGET_BYTES = 128 * 1024 * 1024


@dataclass
class _ProgramEntry:
    program: Any
    vao: Any


class GPUExecutor:
    """Serializes GL work on one persistent thread and reuses size-bound targets."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any] | None, Future[Any] | None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="coldcamera-gpu", daemon=True)
        self._started = False
        self._closed = False
        self._state_lock = threading.Lock()
        self._context: moderngl.Context | None = None
        self._vbo: Any = None
        self._programs: dict[str, _ProgramEntry] = {}
        self._size: tuple[int, int] | None = None
        self._textures: list[Any] = []
        self._framebuffers: list[Any] = []
        self._last_error: str | None = None
        self._context_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def context_error(self) -> str | None:
        return self._context_error

    def render(self, image: np.ndarray, passes: Sequence[GPUShaderPass]) -> np.ndarray:
        if not passes:
            return image.copy()
        rgba = self._to_rgba(image)
        rendered = self._call(lambda: self._render_on_thread(rgba, passes))
        if image.shape[2] == 3:
            return np.ascontiguousarray(rendered[:, :, :3])
        return rendered

    def _call(self, function: Callable[[], Any]) -> Any:
        future: Future[Any] | None = None
        run_inline = False
        with self._state_lock:
            if self._closed:
                raise RuntimeError("GPU executor is closed")
            if self._context_error is not None:
                raise RuntimeError(self._context_error)
            if not self._started:
                self._thread.start()
                self._started = True
            run_inline = threading.current_thread() is self._thread
            if not run_inline:
                future = Future()
                # Keep queue ordering atomic with shutdown: accepted work must
                # always run before the stop sentinel.
                self._jobs.put((function, future))
        if run_inline:
            return function()
        assert future is not None
        return future.result()

    def _run(self) -> None:
        while True:
            function, future = self._jobs.get()
            if function is None:
                break
            assert future is not None
            try:
                result = function()
                self._last_error = None
                future.set_result(result)
            except BaseException as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self._context is None:
                    self._context_error = self._last_error
                future.set_exception(exc)
        self._release_on_thread()

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"GPU processing expects RGB/RGBA, got shape {image.shape}")
        source = np.ascontiguousarray(image, dtype=np.uint8)
        if source.shape[2] == 4:
            return source
        alpha = np.full((*source.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((source, alpha), axis=2)

    def _ensure_context(self) -> moderngl.Context:
        if self._context is None:
            self._context = moderngl.create_standalone_context()
            vertices = np.array(
                [
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    -1.0, 1.0, 0.0, 1.0,
                    -1.0, 1.0, 0.0, 1.0,
                    1.0, -1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0, 1.0,
                ],
                dtype="f4",
            )
            self._vbo = self._context.buffer(vertices.tobytes())
        return self._context

    def _ensure_targets(self, width: int, height: int) -> None:
        ctx = self._ensure_context()
        if self._size == (width, height):
            return
        self._release_targets()
        max_texture_size = int(ctx.info.get("GL_MAX_TEXTURE_SIZE", 16384))
        self._validate_render_target_size(width, height, max_texture_size)
        self._textures = [ctx.texture((width, height), 4, dtype="f1") for _ in range(2)]
        for texture in self._textures:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
        self._framebuffers = [ctx.framebuffer(color_attachments=[texture]) for texture in self._textures]
        self._size = (width, height)

    @staticmethod
    def _validate_render_target_size(width: int, height: int, max_texture_size: int) -> int:
        """Reject target sizes that exceed device limits or the app VRAM budget."""

        if width <= 0 or height <= 0 or width > max_texture_size or height > max_texture_size:
            raise MemoryError(f"GPU target {width}x{height} exceeds GL_MAX_TEXTURE_SIZE={max_texture_size}")
        target_bytes = width * height * 4 * 2
        if target_bytes > GPU_RENDER_TARGET_BUDGET_BYTES:
            raise MemoryError(
                f"GPU targets need {target_bytes} bytes, above the {GPU_RENDER_TARGET_BUDGET_BYTES}-byte render-target budget"
            )
        return target_bytes

    def _compile(self, fragment_shader: str) -> _ProgramEntry:
        entry = self._programs.get(fragment_shader)
        if entry is not None:
            return entry
        ctx = self._ensure_context()
        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader)
        vao = ctx.simple_vertex_array(program, self._vbo, "in_vert", "in_uv")
        entry = _ProgramEntry(program, vao)
        self._programs[fragment_shader] = entry
        return entry

    @staticmethod
    def _prepare_passes(passes: Sequence[GPUShaderPass]) -> list[GPUShaderPass]:
        prepared: list[GPUShaderPass] = []
        index = 0
        while index < len(passes):
            current = passes[index]
            if current.expression is None or not current.pointwise:
                prepared.append(current)
                index += 1
                continue

            expressions: list[str] = []
            uniform_values: dict[str, float | int | bool] = {}
            while index < len(passes) and passes[index].expression is not None and passes[index].pointwise:
                item = passes[index]
                assert item.expression is not None
                expression = item.expression
                for name, value in item.uniforms.items():
                    unique_name = f"{name}_{index}"
                    expression = expression.replace(name, unique_name)
                    uniform_values[unique_name] = value
                expressions.append(expression + "\ncolor = floor(clamp(color, 0.0, 1.0) * 255.0 + 0.000001) / 255.0;")
                index += 1

            declarations = "\n".join(f"uniform float {name};" for name in uniform_values)
            body = "\n".join(expressions)
            source = f"""
            #version 330 core
            uniform sampler2D tDiffuse;
            {declarations}
            in vec2 vUv;
            out vec4 fragColor;
            void main() {{
                vec4 color = texture(tDiffuse, vUv);
                {body}
                fragColor = clamp(color, 0.0, 1.0);
            }}
            """
            prepared.append(GPUShaderPass(uniforms=uniform_values, fragment_shader=source))
        return prepared

    def _render_on_thread(self, image: np.ndarray, passes: Sequence[GPUShaderPass]) -> np.ndarray:
        ctx = self._ensure_context()
        height, width = image.shape[:2]
        self._ensure_targets(width, height)
        prepared = self._prepare_passes(passes)
        self._textures[0].write(image.tobytes(), alignment=1)
        self._textures[0].use(location=0)
        source_index = 0

        for shader_pass in prepared:
            fragment_shader = shader_pass.fragment_shader
            if fragment_shader is None:
                raise ValueError("GPU pass must provide either an expression or a fragment shader")
            entry = self._compile(fragment_shader)
            target_index = 1 - source_index
            self._framebuffers[target_index].use()
            ctx.viewport = (0, 0, width, height)
            self._textures[source_index].use(location=0)
            if "tDiffuse" in entry.program:
                entry.program["tDiffuse"].value = 0
            for name, value in shader_pass.uniforms.items():
                if name in entry.program:
                    entry.program[name].value = int(value) if isinstance(value, bool) else value
            entry.vao.render(mode=moderngl.TRIANGLES)
            source_index = target_index

        raw = self._framebuffers[source_index].read(components=4, alignment=1)
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4)).copy()

    def _release_targets(self) -> None:
        for framebuffer in self._framebuffers:
            framebuffer.release()
        for texture in self._textures:
            texture.release()
        self._framebuffers.clear()
        self._textures.clear()
        self._size = None

    def _release_on_thread(self) -> None:
        self._release_targets()
        for entry in self._programs.values():
            entry.vao.release()
            entry.program.release()
        self._programs.clear()
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        if self._context is not None:
            self._context.release()
            self._context = None

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if not started:
            return
        self._jobs.put((None, None))
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(0.0, timeout))


_gpu_executor: GPUExecutor | None = None
_gpu_executor_lock = threading.Lock()


def get_gpu_executor() -> GPUExecutor:
    global _gpu_executor
    with _gpu_executor_lock:
        if _gpu_executor is None or _gpu_executor._closed:
            _gpu_executor = GPUExecutor()
        return _gpu_executor


def peek_gpu_executor() -> GPUExecutor | None:
    """Return the existing executor without creating its worker or GL context."""

    return _gpu_executor


def shutdown_gpu_executor(timeout: float = 2.0) -> None:
    global _gpu_executor
    with _gpu_executor_lock:
        executor = _gpu_executor
        _gpu_executor = None
    if executor is not None:
        executor.shutdown(timeout=timeout)

\n
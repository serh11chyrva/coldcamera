from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from coldcamera.classes.effect import GPUShaderPass
from coldcamera.core.gpu import get_gpu_executor


class ShaderProcessorBase(ABC):
    """Small effect-facing adapter over the application's shared GL executor."""

    @abstractmethod
    def fragment_shader(self) -> str:
        """Return the fragment shader source code as a string."""

    def get_uniforms(self, **kwargs) -> dict[str, float | int | bool]:
        """Convert effect parameters to GL uniform values."""

        return kwargs

    def process(self, image, **kwargs) -> np.ndarray:
        """Apply the shader using the persistent, thread-affine GPU context."""

        if isinstance(image, Image.Image):
            img_data = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"Shader input must be RGB/RGBA, got shape {image.shape}")
            source = np.ascontiguousarray(image, dtype=np.uint8)
            if source.shape[2] == 3:
                alpha = np.full((*source.shape[:2], 1), 255, dtype=np.uint8)
                img_data = np.concatenate((source, alpha), axis=-1)
            else:
                img_data = source
        else:
            raise TypeError("ShaderProcessorBase.process() accepts only PIL.Image or numpy.ndarray")

        shader_pass = GPUShaderPass(
            uniforms=self.get_uniforms(**kwargs),
            fragment_shader=self.fragment_shader(),
            pointwise=False,
        )
        return get_gpu_executor().render(img_data, [shader_pass])

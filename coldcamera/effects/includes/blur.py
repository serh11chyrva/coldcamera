import blend_modes as bm
import cv2
import numpy as np

from coldcamera.classes.effect import EffectBase, EffectCapabilities
from coldcamera.core.processing_settings import ProcessingBackend
from coldcamera.classes.parameter import EffectParam
from coldcamera.classes.shader_processor import ShaderProcessorBase
from coldcamera.types import Processable
from coldcamera.utils.add_alpha_channel import add_alpha_channel


class BlurShaderProcessor(ShaderProcessorBase):
    author: str = "deathloveleopards"

    def fragment_shader(self) -> str:
        return """
        #version 330 core
        uniform sampler2D tDiffuse;
        uniform float amount;
        uniform float angle;
        in vec2 vUv;
        out vec4 fragColor;

        const int NUM_SAMPLES = 32;

        void main() {
            vec2 texSize = textureSize(tDiffuse, 0);
            vec2 dir = vec2(cos(angle), sin(angle)) / texSize;

            vec4 sum = vec4(0.0);
            float halfRange = amount * 0.5;

            for (int i = 0; i < NUM_SAMPLES; i++) {
                float t = float(i) / float(NUM_SAMPLES - 1);
                float offset = mix(-halfRange, halfRange, t);
                sum += texture(tDiffuse, vUv + dir * offset);
            }

            fragColor = sum / float(NUM_SAMPLES);
        }
        """

    def set_uniforms(self, **kwargs):
        return None

    def get_uniforms(self, **kwargs) -> dict[str, float]:
        return {"amount": float(kwargs.get("amount", 0.0)), "angle": float(kwargs.get("angle", 0.0))}


class BlurEffect(EffectBase):
    def __init__(self, name="Blur"):
        super().__init__(
            name,
            params=[
                EffectParam("amount", float, 0.0, default=0.0),
                EffectParam("angle", float, 0.0, default=0.0),
                EffectParam("opacity", float, 1.0, default=1.0),
                EffectParam("blend_mode", str, "lighten_only", default="lighten_only"),
            ],
        )
        # OpenGL resources belong to the processing thread, not the GUI that
        # may construct an effect for its parameter editor.
        self.processor: BlurShaderProcessor | None = None

    def apply(self, input_data: Processable) -> Processable:
        return self.apply_with_backend(input_data, ProcessingBackend.AUTO)

    def apply_with_backend(self, input_data: Processable, backend: ProcessingBackend) -> Processable:
        img_rgb = np.array(input_data).astype(np.uint8)

        if self.get_parameter("amount") <= 0:
            return img_rgb

        if backend == ProcessingBackend.CPU:
            blurred = self._blur_cpu(img_rgb).astype(np.float32)
        else:
            try:
                if self.processor is None:
                    self.processor = BlurShaderProcessor()
                blurred = self.processor.process(
                    img_rgb,
                    amount=self.get_parameter("amount"),
                    angle=np.radians(self.get_parameter("angle")),
                ).astype(np.float32)
            except Exception:
                blurred = self._blur_cpu(img_rgb).astype(np.float32)

        base = add_alpha_channel(img_rgb).astype(np.float32)

        base /= 255.0
        blurred /= 255.0

        blend_func = getattr(bm, self.get_parameter("blend_mode"), bm.normal)
        blended = blend_func(base, blurred, self.get_parameter("opacity"))

        return (blended * 255).astype(np.uint8)

    def get_execution_capabilities(self) -> EffectCapabilities:
        radius = int(np.ceil(float(self.get_parameter("amount")) * 0.5 + 1.0))
        return EffectCapabilities(locality="local", halo_x=radius, halo_y=radius, supported_backends=frozenset({"cpu", "gpu"}))

    def supports_gpu(self) -> bool:
        return True

    def _blur_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU fallback that matches the shader's 32 linear samples."""

        height, width = image.shape[:2]
        amount = float(self.get_parameter("amount"))
        angle = float(np.radians(self.get_parameter("angle")))
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        direction_x, direction_y = np.cos(angle), np.sin(angle)
        result = np.zeros_like(image, dtype=np.float32)
        for sample in range(32):
            t = sample / 31.0
            offset = (t - 0.5) * amount
            map_x = np.asarray(xx + direction_x * offset, dtype=np.float32)
            map_y = np.asarray(yy - direction_y * offset, dtype=np.float32)
            result += cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result / 32.0

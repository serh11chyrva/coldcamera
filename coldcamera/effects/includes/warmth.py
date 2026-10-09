import numpy as np

from coldcamera.classes.effect import EffectBase, EffectCapabilities, GPUShaderPass
from coldcamera.classes.parameter import EffectParam
from coldcamera.types import Processable


class WarmthEffect(EffectBase):
    author: str = "deathloveleopards"

    def __init__(self, name="Warmth"):
        super().__init__(
            name,
            params=[EffectParam("warmth", float, 0.0, default=0.0)],
        )

    def apply(self, input_data: Processable) -> Processable:
        img = np.array(input_data).astype(np.float32)
        warmth_val = self.get_parameter("warmth") / 100.0
        img[..., 0] *= 1.0 + warmth_val  # reduce blue
        img[..., 2] *= 1.0 - warmth_val  # increase red
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def get_execution_capabilities(self) -> EffectCapabilities:
        return EffectCapabilities(locality="pointwise", supported_backends=frozenset({"cpu", "gpu"}))

    def get_gpu_pass(self) -> GPUShaderPass:
        warmth = float(self.get_parameter("warmth")) / 100.0
        return GPUShaderPass(
            uniforms={"uWarmth": warmth},
            expression="color.r = clamp(color.r * (1.0 + uWarmth), 0.0, 1.0); color.b = clamp(color.b * (1.0 - uWarmth), 0.0, 1.0);",
        )

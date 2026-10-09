import numpy as np

from coldcamera.classes.effect import EffectBase, EffectCapabilities, GPUShaderPass
from coldcamera.classes.parameter import EffectParam
from coldcamera.types import Processable


class ContrastBrightnessEffect(EffectBase):
    author: str = "deathloveleopards"

    def __init__(self, name="Contrast/Brightness"):
        super().__init__(
            name,
            params=[
                EffectParam("contrast", float, 1.0, default=1.0),
                EffectParam("brightness", float, 0.0, default=0.0),
            ],
        )

    def apply(self, input_data: Processable) -> Processable:
        img = np.array(input_data).astype(np.float32)
        contrast = self.get_parameter("contrast")
        brightness = self.get_parameter("brightness")
        img = (img - 127.5) * contrast + 127.5 + brightness
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def get_execution_capabilities(self) -> EffectCapabilities:
        return EffectCapabilities(locality="pointwise", supported_backends=frozenset({"cpu", "gpu"}))

    def get_gpu_pass(self) -> GPUShaderPass:
        return GPUShaderPass(
            uniforms={
                "uContrast": float(self.get_parameter("contrast")),
                "uBrightness": float(self.get_parameter("brightness")) / 255.0,
            },
            expression="color = clamp((color - vec4(0.5)) * uContrast + vec4(0.5 + uBrightness), 0.0, 1.0);",
        )

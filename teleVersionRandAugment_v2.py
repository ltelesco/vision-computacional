import torch
import math

from torchvision.transforms.autoaugment import RandAugment
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
from torchvision import transforms as T

def _apply_op(img: Tensor, 
                  op_name: str,
                  magnitude: float,
                  interpolation: InterpolationMode,
                  fill: Optional[List[float]]):
        #print("op name es:", op_name)          
        if op_name == "ShearX":
            # magnitude should be arctan(magnitude)
            # official autoaug: (1, level, 0, 0, 1, 0)
            # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
            # compared to
            # torchvision:      (1, tan(level), 0, 0, 1, 0)
            # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif op_name == "ShearY":
            # magnitude should be arctan(magnitude)
            # See above
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif op_name == "TranslateX":
            img = F.affine(
                img,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "TranslateY":
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "Rotate": #toDo
            img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
        elif op_name == "Brightness":
            img = F.adjust_brightness(img, 1.0 + magnitude)
        elif op_name == "Color": #toDo - listo!
            jitter = T.ColorJitter(hue=magnitude)
            img = jitter(img)
        elif op_name == "Contrast":
            img = F.adjust_contrast(img, 1.0 + magnitude)
        elif op_name == "Sharpness":
            img = F.adjust_sharpness(img, 1.0 + magnitude)
        elif op_name == "Posterize":
            img = F.posterize(img, int(magnitude))
        elif op_name == "Solarize":
            img = F.solarize(img, magnitude)
        elif op_name == "AutoContrast":
            img = F.autocontrast(img)
        elif op_name == "Equalize":
            img = F.equalize(img)
        elif op_name == "Invert":
            img = F.invert(img)
        elif op_name == "HorizontalFlip": #toDo
            hflipper = T.RandomHorizontalFlip(p=1)
            img = hflipper(img)
        elif op_name == "VerticalFlip": #toDo
            vflipper = T.RandomVerticalFlip(p=1)
            img = vflipper(img) 
        elif op_name == "Perspective": #toDo
            rPerspective = T.RandomPerspective(distortion_scale=magnitude, p=1.0)
            img = rPerspective(img)
        elif op_name == "Saturation": #toDo listo!
            img = T.functional.adjust_saturation(img, 1.0 + magnitude)
        elif op_name == "Identity": #toDo - Queda como esta
            pass
        else:
            raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img


class TeleVersionRandAugment(RandAugment):

    def __init__(
    self,
    num_ops: int = 2,
    magnitude: int = 9,
    num_magnitude_bins: int = 31,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    filters = None,
    ) -> None:
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
        self.filters = filters

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return self.filters
    
    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = 3, 3, 3
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img
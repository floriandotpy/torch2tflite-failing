import time
from pathlib import Path
from unittest import mock
from typing import List, Tuple, Dict, Optional

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch import Tensor
from torchvision.models.detection.transform import _get_shape_onnx
from torchvision.models.detection.transform import _fake_cast_onnx


def _custom_resize_image_and_masks(image: Tensor, self_min_size: float, self_max_size: float,
                                   target: Optional[Dict[str, Tensor]] = None,
                                   fixed_size: Optional[Tuple[int, int]] = None,
                                   ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    """
    Identical to: torchvision.models.detection.transform._resize_image_and_masks (torchvision version 0.10.0+cu102)
    With only one difference: `align_corners=True` instead of False

    This avoids the following error (which woould show up in the step onnx -> tensorflow)
    ```
    RuntimeError: Resize coordinate_transformation_mode=pytorch_half_pixel is not supported in Tensorflow.
    ```
    """
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(image[None],
                                            size=size,
                                            scale_factor=scale_factor,
                                            mode='bilinear',
                                            recompute_scale_factor=recompute_scale_factor,
                                            align_corners=True  # <-- this is our only change (`False` in the original)
                                            )[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor,
                                               recompute_scale_factor=recompute_scale_factor)[:, 0].byte()
        target["masks"] = mask
    return image, target


def main():
    t_start = time.time()
    assets_dir = Path('assets')
    if not assets_dir.exists():
        assets_dir.mkdir(exist_ok=True)
    fname_onnx = assets_dir / 'model.onnx'

    print("Loading pytorch model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # dummy input
    x = torch.zeros([1, 3, 256, 256])

    print("Converting to onnx...")
    with mock.patch('torchvision.models.detection.transform._resize_image_and_masks', _custom_resize_image_and_masks):
        torch.onnx.export(
            model,
            x,
            fname_onnx,
            export_params=True,
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # variable length axes
                'output': {0: 'batch_size'}
            }
        )
    print(f"[onnx] Written {fname_onnx}")
    duration = int(time.time() - t_start)
    print(f"Time elapsed: {duration} seconds")


if __name__ == '__main__':
    main()

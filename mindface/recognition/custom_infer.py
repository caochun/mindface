"""
inference of face recognition models.
"""
import os
import argparse
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

def infer(img, model):
    """
    The inference of arcface.

    Args:
        img (NumPy): The input image.
        model: The recognition model.

    Examples:
        >>> img = input_img
        >>> model = get_mbf(num_features=512)
        >>> out1 = infer(img, model)
    """
    assert (img.shape[-1] == 112 and img.shape[-2] == 112)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    img = ((img / 255) - 0.5) / 0.5
    img = ms.Tensor(img, ms.float32)
    if len(img.shape) == 4:
        pass
    elif len(img.shape) == 3:
        img = img.expand_dims(axis=0)
    net_out = model(img)
    embeddings = net_out.asnumpy()

    return embeddings
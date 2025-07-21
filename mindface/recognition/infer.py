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
from .models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l


def infer(img, backbone="iresnet50", num_features=512, pretrained=False):
    """
    The inference of arcface.

    Args:
        img (NumPy): The input image.
        backbone (Object): Arcface model without loss function. Default: "iresnet50".
        pretrained (Bool): Pretrain. Default: False.

    Examples:
        >>> img = input_img
        >>> out1 = infer(input_img, backbone="iresnet50",
                        pretrained="/path/to/eval/ArcFace.ckpt")
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    assert (img.shape[-1] == 112 and img.shape[-2] == 112)
    img = ((img / 255) - 0.5) / 0.5
    img = ms.Tensor(img, ms.float32)
    if len(img.shape) == 4:
        pass
    elif len(img.shape) == 3:
        img = img.expand_dims(axis=0)

    if backbone == 'iresnet50':
        model = iresnet50(num_features=num_features)
        print("Finish loading iresnet50")
    elif backbone == 'iresnet100':
        model = iresnet100(num_features=num_features)
        print("Finish loading iresnet100")
    elif backbone == 'mobilefacenet':
        model = get_mbf(num_features=num_features)
        print("Finish loading mobilefacenet")
    elif backbone == 'vit_t':
        model = vit_t(num_features=num_features)
        print("Finish loading vit_t")
    elif backbone == 'vit_s':
        model = vit_s(num_features=num_features)
        print("Finish loading vit_s")
    elif backbone == 'vit_b':
        model = vit_b(num_features=num_features)
        print("Finish loading vit_b")
    elif backbone == 'vit_l':
        model = vit_l(num_features=num_features)
        print("Finish loading vit_l")
    else:
        raise NotImplementedError

    if pretrained:
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(model, param_dict)

    net_out = model(img)
    embeddings = net_out.asnumpy()

    return embeddings


def preprocess_image(img_path):
    """Load and preprocess image to tensor format, resize to 112x112"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = Image.open(img_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((112, 112), Image.BICUBIC)
    img = np.array(img).transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = ms.Tensor(img, ms.float32)
    img = img.expand_dims(0)
    
    print(f"Preprocessed image shape: {img.shape}")
    return img


def run_inference(img_path, backbone="iresnet50", num_features=512, pretrained=None):
    """Execute face feature extraction inference"""
    print(f"Processing image: {img_path}")
    print(f"Model: {backbone}, Features: {num_features}")
    print(f"Pretrained weights: {pretrained if pretrained else 'default'}")
    
    img_tensor = preprocess_image(img_path)
    
    print("Extracting features...")
    feature = infer(
        img=img_tensor,
        backbone=backbone,
        num_features=num_features,
        pretrained=pretrained if pretrained else False
    )
    
    print(f"Feature extraction completed, shape: {feature.shape}")
    return feature


def main():
    parser = argparse.ArgumentParser(description="MindFace inference script")
    parser.add_argument("--backbone", type=str, default="iresnet50",
                        choices=["iresnet50", "iresnet100", "mobilefacenet"],
                        help="Model backbone")
    parser.add_argument("--num_features", type=int, default=512, help="Feature dimension")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained model path")
    parser.add_argument("--img_path", type=str, required=True, help="Input image path")
    parser.add_argument("--output_path", type=str, default=None, help="Feature save path")
    
    args = parser.parse_args()
    
    try:
        feature = run_inference(
            img_path=args.img_path,
            backbone=args.backbone,
            num_features=args.num_features,
            pretrained=args.checkpoint
        )
        
        print(f"\n=== Results ===")
        print(f"Feature shape: {feature.shape}")
        print(f"Feature norm: {np.linalg.norm(feature.asnumpy()):.6f}")
        
        if args.output_path:
            feature_np = feature.asnumpy()
            np.save(args.output_path, feature_np)
            print(f"Features saved to: {args.output_path}")
        
        print("Inference completed!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
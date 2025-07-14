import argparse
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint

from models import RetinaFace, resnet50, mobilenet025
from runner import DetectionEngine, read_yaml

def modify_param_keys(checkpoint_path, output_path):
    """将预训练模型参数key中的所有字母转换为小写"""
    
    # 加载原始checkpoint
    param_dict = load_checkpoint(checkpoint_path)
    print(f"Original checkpoint loaded from: {checkpoint_path}")
    print(f"Original parameter count: {len(param_dict)}")
    
    # 创建新的参数字典
    new_param_dict = {}
    
    modified_count = 0
    
    # 遍历所有参数，将key转换为小写
    for old_key, param in param_dict.items():
        new_key = old_key.lower()
        
        if new_key != old_key:
            modified_count += 1
            print(f"Modified: {old_key} -> {new_key}")
        
        new_param_dict[new_key] = param
    
    print(f"Total modified keys: {modified_count}")
    print(f"New parameter count: {len(new_param_dict)}")
    
    # 保存修复后的checkpoint
    save_checkpoint(new_param_dict, output_path)
    print(f"Modified checkpoint saved to: {output_path}")
    
    return new_param_dict
    
def main():
    parser = argparse.ArgumentParser(description='修复预训练模型参数key')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='原始checkpoint路径')
    parser.add_argument('--output_checkpoint', type=str, required=True,
    help='修复后checkpoint保存路径')
    parser.add_argument('--config', default='mindface/detection/configs/RetinaFace_mobilenet025.yaml', 
                        type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 修复参数key
    print("开始修复参数key...")
    modified_param_dict = modify_param_keys(args.input_checkpoint, args.output_checkpoint)
    
if __name__ == '__main__':
    main()
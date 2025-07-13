import argparse
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint

from models import RetinaFace, resnet50, mobilenet025
from runner import DetectionEngine, read_yaml

def modify_param_keys(checkpoint_path, output_path):
    """Convert all letters in pretrained model parameter keys to lowercase"""
    
    # Load original checkpoint
    param_dict = load_checkpoint(checkpoint_path)
    print(f"Original checkpoint loaded from: {checkpoint_path}")
    print(f"Original parameter count: {len(param_dict)}")
    
    # Create new parameter dictionary
    new_param_dict = {}
    
    modified_count = 0
    
    # Iterate through all parameters, convert keys to lowercase
    for old_key, param in param_dict.items():
        new_key = old_key.lower()
        
        if new_key != old_key:
            modified_count += 1
            print(f"Modified: {old_key} -> {new_key}")
        
        new_param_dict[new_key] = param
    
    print(f"Total modified keys: {modified_count}")
    print(f"New parameter count: {len(new_param_dict)}")
    
    # Save the fixed checkpoint
    save_checkpoint(new_param_dict, output_path)
    print(f"Modified checkpoint saved to: {output_path}")
    
    return new_param_dict
    
def main():
    parser = argparse.ArgumentParser(description='Fix pretrained model parameter keys')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='Original checkpoint path')
    parser.add_argument('--output_checkpoint', type=str, required=True,
                        help='Fixed checkpoint save path')
    
    args = parser.parse_args()
    
    # Fix parameter keys
    print("Starting to fix parameter keys...")
    modified_param_dict = modify_param_keys(args.input_checkpoint, args.output_checkpoint)
    
if __name__ == '__main__':
    main()
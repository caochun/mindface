# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Eval Retinaface_resnet50_or_mobilenet0.25."""
import argparse
import numpy as np
import cv2
import os
import base64

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .utils import prior_box
from .models import RetinaFace, resnet50, mobilenet025
from .runner import DetectionEngine, read_yaml

def infer(cfg, network, return_location=False):
    """test one image"""
    # testing image

    conf_test = cfg['conf']
    image_path = cfg['image_path']


    detection = DetectionEngine(nms_thresh = cfg['val_nms_threshold'], conf_thresh = cfg['val_confidence_threshold'],
                                    iou_thresh = cfg['val_iou_threshold'], var = cfg['variance'])

    # testing begin
    print('Predict box starting')
    print(image_path)
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    target_size = 1600
    max_size = 2176
    priors = prior_box(image_sizes=(max_size, max_size),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)

    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)

    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    assert img.shape[0] <= max_size and img.shape[1] <= max_size
    image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
    image_t[:, :] = (104.0, 117.0, 123.0)
    image_t[0:img.shape[0], 0:img.shape[1]] = img
    img = image_t

    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = Tensor(img)

    boxes, confs, _ = network(img)
    boxes = detection.infer(boxes, confs, resize, scale, priors)
    img_each = cv2.imread(image_path, cv2.IMREAD_COLOR)

    H, W = img_each.shape[:2]
    crop_count = 1

    results = []
    for box in boxes:
        confidence = box[4]
        if confidence > conf_test:
            x, y, w, h= map(int, box[:4])

            # 边界裁剪，防止越界
            x = max(0, x)
            y = max(0, y)
            x2 = min(W, x + w)
            y2 = min(H, y + h)

            actual_width = x2 - x
            actual_height = y2 - y
            crop = img_each[y:y2, x:x2]

            crop_path = f"{os.path.basename(image_path).split('.')[0]}_crop_{crop_count}.jpg"
            cv2.imwrite(crop_path, crop)
            print(f"[✓] Cropped face saved: {crop_path}")
            
            if return_location:
                _, buffer = cv2.imencode('.jpg', crop)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
                results.append({
                    'crop_path': crop_path,
                    'location': {
                        'x': x,
                        'y': y,
                        'width': actual_width,
                        'height': actual_height,
                        'localface': face_base64,
                        'confidence': float(confidence)
                    }
                })
            else:
                results.append(crop_path)

            crop_count += 1
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--config', default='/home/fangwy/mindface/mindface/detection/configs/RetinaFace_mobilenet025.yaml', type=str,
                        help='configs path')
    parser.add_argument('--checkpoint', type=str, default='/home/fangwy/pretrained/Fix_RetinaFace_MobileNet025.ckpt',
                        help='checpoint path')
    parser.add_argument('--image_path', type=str, default='/home/fangwy/mindface/test/detection/imgs/0000.jpg',
                        help='image path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence of bbox')
    args = parser.parse_args()

    config = read_yaml(args.config)
    if args.image_path:
        config['image_path'] = args.image_path
    if args.conf:
        config['conf'] = args.conf
    if args.checkpoint:
        config['val_model'] = args.checkpoint
    infer(cfg=config)
import argparse
import cv2
import os
import base64
import sys
from datetime import datetime
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.custom_infer import infer as detection_infer
from detection.models import RetinaFace, mobilenet025, resnet50
from recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l
from recognition.custom_infer import infer as recognition_infer
from detection.runner import read_yaml
from es import FaceEmbeddingES
import numpy as np
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detection_network, recognition_network = None, None
detection_config, recognition_config = None, None

def save_base64_to_jpg(base64_str, output_path):
    """
    将Base64字符串保存为JPG图片
    :param base64_str: Base64编码的图片字符串
    :param output_path: 输出图片路径（包含.jpg后缀）
    """
    try:
        # 移除可能的Base64前缀（如"data:image/jpeg;base64,"）
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # 解码Base64字符串
        img_data = base64.b64decode(base64_str)
        
        # 写入文件
        with open(output_path, "wb") as f:
            f.write(img_data)
        
        print(f"图片已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"保存失败: {str(e)}")
        return False

def get_embedding(image_path):
    detection_config['image_path'] = image_path
    recognition_config = {}
    
    face_img_paths = detection_infer(detection_config, detection_network)
    if len(face_img_paths) == 0:
        return None
    batch_images = []
    for face_img_path in face_img_paths:
        image_np = cv2.imread(face_img_path)
        image_np = cv2.resize(image_np, (112, 112))
        image_np = np.transpose(image_np, (2, 0, 1))
        batch_images.append(image_np)
        if os.path.exists(face_img_path):
            os.remove(face_img_path)
    
    if batch_images:
        batch_array = np.stack(batch_images, axis=0)
    embeddings = recognition_infer(image_np,recognition_network)
    return embeddings

# 创建人脸库
@app.route('/create', methods=['POST'])
def create_face_store():
    json_list = request.get_json()
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    reponame = params['reponame']
    code = face_es.create_face_store(reponame)
    if code.value == 0:
        return jsonify({
            'status': 'success',
            'Code': 0,
            'Message': f'Success Create {reponame}'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'Code': code.value,
            'Message': f'Fail Create {reponame}'
        }), 404


# 注册人脸
@app.route('/register', methods=['POST'])
def register_face():
    json_list = request.get_json()
    if 'img' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1002,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    image = json_list['img']
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']

    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image, temp_path)

    embeddings = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if embeddings is None:
        return jsonify({
            'status': 'error',
            'Code': 1008,
            'Message': '注册失败！人脸质量差'
        }), 404
    code = face_es.register_face(identity, embeddings[0], reponame)
    if code.value == 0:
        return jsonify({
            'status': 'success',
            'Code': 0,
            'Message': f'ID: {identity} Register to {reponame}'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'Code': code.value,
            'Message': f'ID: {identity} Fail to Register to {reponame}'
        }), 404

# 注销人脸
@app.route('/delete', methods=['POST'])
def delete_face():
    json_list = request.get_json()
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']
    code = face_es.delete_face(identity, reponame)
    if code.value == 0:
        return jsonify({
            'status': 'success',
            'Code': 0,
            'Message': f'Success! Revoke ID: {identity} In {reponame}'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'Code': code.value,
            'Message': f'Fail!'
        }), 404

# 更新人脸
@app.route('/update', methods=['POST'])
def update_face():
    json_list = request.get_json()
    if 'img' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1002,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    image = json_list['img']
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']

    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)

    embeddings = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if embeddings is None:
        return jsonify({
            'status': 'error',
            'Code': 1008,
            'Message': '更新失败！人脸质量差'
        }), 404
    code = face_es.update_face(identity, embeddings[0], reponame)
    if code.value == 0:
        return jsonify({
            'status': 'success',
            'Code': 0,
            'Message': f'更新成功'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'Code': code.value,
            'Message': f'更新失败'
        }), 404

# 识别人脸
@app.route('/recognize', methods=['POST'])
def recognize_face():
    json_list = request.get_json()
    if 'img' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1002,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    image = json_list['img']
    threshold = 0.9
    identity = params['id']
    reponame = params['reponame']
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)
    
    query_embs = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    for query_emb in query_embs:
        code, results = face_es.search_similar_faces(query_emb, reponame, 1)
        if code.value == 0 and results[0]['id'] == identity:
            return jsonify({
                'status': 'success',
                'Code': 0,
                'Confidence': results[0]['similarity'],
                'Message': 'Success'
            }), 200
    return jsonify({
            'status': 'error',
            'Code': -1,
            'Message': 'Fail to Recognize Face'
        }), 404

# 识别多个人脸
@app.route('/recognizeN', methods=['POST'])
def recognizeN_face():
    json_list = request.get_json()
    if 'img' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1002,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': 1001,
            'Message': '缺少参数reponame'
        }), 404
    image = json_list['img']
    threshold = 0.9
    identity = params['id']
    reponame = params['reponame']
    
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)
    
    query_embs = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    response = []
    for query_emb in query_embs:
        code, results = face_es.search_similar_faces(query_emb, reponame, 1)
        candidate = {}
        if code.value == 0:
            candidate['Confidence'] = results[0]['similarity']
            candidate['Id'] = results[0]['id']
        else:
            return jsonify({
                    'status': 'error',
                    'Code': code.value,
                    'Message': 'Fail to recognize face'
                }), 404
        res = {}
        res['Candidates'] = []
        res['Candidates'].append(candidate)
        res['CandidatesCount'] = 1
        res['CandidatesMessage'] = ""
        res['Feature'] = ""
        location = {}
        location['Confidence'] = ''
        location['Height'] = ''
        location['LocalFace'] = ''
        location['Width'] = ''
        location['X'] = ''
        location['Y'] = ''
        res['Location'] = location
        response.append(res)
    return jsonify({
            'status': 'success',
            'Code': 0,
            'Face': response,
            'Ignore': None
        }), 200
  

def init_model():
    if detection_config['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=detection_config['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = detection_config['device_target'])

    if detection_config['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif detection_config['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    global detection_network
    detection_network = RetinaFace(phase='predict',backbone=backbone,in_channel=detection_config['in_channel'],out_channel=detection_config['out_channel'])
    backbone.set_train(False)
    detection_network.set_train(False)
    # load checkpoint
    assert detection_config['val_model'] is not None, 'val_model is None.'
    param_dict = load_checkpoint(detection_config['val_model'])
    print(f"Load detection model done. {detection_config['val_model']}")
    detection_network.init_parameters_data()
    load_param_into_net(detection_network, param_dict)
    # 准备预热输入（形状需与实际输入一致）
    warmup_input = np.float32(np.random.rand(1, 3, 2176, 2176))
    warmup_input = Tensor(warmup_input)
    # 执行预热（通常3-5次）
    for _ in range(1):
        detection_network(warmup_input)
    
    global recognition_network
    if recognition_config["backbone"] == 'mobilefacenet':
        recognition_network = get_mbf(num_features=512)
        print("Finish loading mobilefacenet")

    if recognition_config["pretrained"]:
        param_dict = load_checkpoint(recognition_config["pretrained"])
        load_param_into_net(recognition_network, param_dict)
        
    # 准备预热输入（形状需与实际输入一致）
    warmup_input = np.float32(np.random.rand(1,3,112,112))
    warmup_input = Tensor(warmup_input)
    # 执行预热（通常3-5次）
    for _ in range(3):
        recognition_network(warmup_input)
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start server')
    parser.add_argument('--detection_model', default='mobilenet', type=str,
                        help='detection model type')
    parser.add_argument('--recognition_model', type=str, default='mobilefacenet',
                        help='recognition model type')
    args = parser.parse_args()
    
    if args.detection_model == "mobilenet":
        detection_config = read_yaml("detection/configs/RetinaFace_mobilenet025.yaml")
        detection_config['conf'] = 0.9
        detection_config['val_model'] = "/home/fangwy/pretrained/Fix_RetinaFace_MobileNet025.ckpt"
    recognition_config = {}
    if args.recognition_model == "mobilefacenet":
        recognition_config["backbone"] = args.recognition_model
        recognition_config["pretrained"] = "/home/fangwy/pretrained/mobile_casia_ArcFace.ckpt"
    
    init_model()
    
    face_es = FaceEmbeddingES()
    app.run(host='0.0.0.0', port=5666, debug=True)
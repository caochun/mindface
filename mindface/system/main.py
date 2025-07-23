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
from milvus import FaceEmbeddingMilvus
from running_code import RunningCode
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

def get_embedding(image_path, return_location=False):
    detection_config['image_path'] = image_path
    
    face_results = detection_infer(detection_config, detection_network, return_location)
    if len(face_results) == 0:
        return []
    batch_images = []
    face_locations = []
    for face_result in face_results:
        if return_location:
            face_img_path = face_result['crop_path']
            face_locations.append(face_result['location'])
        else:
            face_img_path = face_result
        image_np = cv2.imread(face_img_path)
        image_np = cv2.resize(image_np, (112, 112))
        image_np = np.transpose(image_np, (2, 0, 1))
        batch_images.append(image_np)
        if os.path.exists(face_img_path):
            os.remove(face_img_path)
    
    if batch_images:
        batch_array = np.stack(batch_images, axis=0)
    embeddings = recognition_infer(batch_array,recognition_network)
    
    if return_location:
        results = []
        for i, embedding in enumerate(embeddings):
            results.append({
                'embedding': embedding,
                'location':face_locations[i]
            })
        return results
    else:            
        return embeddings

# 创建人脸库
@app.route('/create', methods=['POST'])
def create_face_store():
    json_list = request.get_json()
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    reponame = params['reponame']
    code = face_db.create_face_store(reponame)
    if code.value == 0:
        return jsonify({
            'status': 'success',
            'Code': code.value,
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
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    image = json_list['img']
    params = json_list['params']
    if params.get('id',None) is None or params['id'] == '':
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None or params['reponame'] == '':
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']

    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    if image == "" or not save_base64_to_jpg(image, temp_path):
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404

    embeddings = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if len(embeddings) == 0:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.POOR_FACE_QUALITY.value,
            'Message': '注册失败！人脸质量差'
        }), 404
    code = face_db.register_face(identity, embeddings[0], reponame)
    if code == RunningCode.SUCCESS:
        return jsonify({
            'status': 'success',
            'Code': code.value,
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
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']
    code = face_db.delete_face(identity, reponame)
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
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    image = json_list['img']
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    identity = params['id']
    reponame = params['reponame']

    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    if image == "" or not save_base64_to_jpg(image, temp_path):
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404

    embeddings = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if len(embeddings) == 0:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.POOR_FACE_QUALITY.value,
            'Message': '更新失败！人脸质量差'
        }), 404
    code = face_db.update_face(identity, embeddings[0], reponame)
    if code == RunningCode.SUCCESS:
        return jsonify({
            'status': 'success',
            'Code': code.value,
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
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('id',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数id'
        }), 404
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    image = json_list['img']
    threshold = 0.9
    identity = params['id']
    reponame = params['reponame']
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    if image == "" or not save_base64_to_jpg(image, temp_path):
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    query_embs = get_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    error_code = -1
    for query_emb in query_embs:
        code, result = face_db.recognize_face(identity, query_emb, reponame, threshold)
        if code == RunningCode.SUCCESS and result['id'] == identity:
            return jsonify({
                'status': 'success',
                'Code': code.value,
                'Confidence': result['similarity'],
                'Message': 'Success'
            }), 200
        elif code == RunningCode.FACE_STORE_NOT_EXISTS:
            error_code = code.value
            break
        elif code == RunningCode.ID_NOT_FOUND:
            error_code = code.value
            break
    if len(query_embs) == 0:
        error_code = RunningCode.NO_FACE_DETECTED.value
    return jsonify({
            'status': 'error',
            'Code': error_code,
            'Message': 'Fail to Recognize Face'
        }), 404

# 识别多个人脸
@app.route('/recognizeN', methods=['POST'])
def recognizeN_face():
    json_list = request.get_json()
    if 'img' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    if 'params' not in json_list:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数'
        }), 404
    params = json_list['params']
    if params.get('reponame',None) is None:
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_PARAMS.value,
            'Message': '缺少参数reponame'
        }), 404
    image = json_list['img']
    threshold = 0.9
    reponame = params['reponame']
    
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    if image == "" or not save_base64_to_jpg(image, temp_path):
        return jsonify({
            'status': 'error',
            'Code': RunningCode.MISSING_FILE.value,
            'Message': '缺少图片'
        }), 404
    
    query_results = get_embedding(temp_path, return_location=True)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    face_store_exists = True
    response = []
    for query_result in query_results:
        query_emb = query_result['embedding']
        query_loc = query_result['location']
        res = {}
        res['Candidates'] = []
        res['CandidatesCount'] = 0
        res['CandidatesMessage'] = ""
        res['Feature'] = ""
        location = {}
        location['Confidence'] = query_loc['confidence']
        location['Height'] = query_loc['height']
        location['LocalFace'] = query_loc['localface']
        location['Width'] = query_loc['width']
        location['X'] = query_loc['x']
        location['Y'] = query_loc['y']
        code, results = face_db.search_similar_faces(query_emb, reponame, k=1, threshold=threshold)
        if code == RunningCode.SUCCESS:
            for result in results:
                candidate = {}
                candidate['Confidence'] = result['similarity']
                candidate['Id'] = result['id']
                res['Candidates'].append(candidate)
                res['CandidatesCount'] += 1
        else:
            if code == RunningCode.FACE_STORE_NOT_EXISTS:
                face_store_exists = False
            res['Feature'] = query_emb.tolist()
        res['Location'] = location
        response.append(res)

    if len(query_results) == 0:
        return jsonify({
                'status': 'error',
                'Code': RunningCode.NO_FACE_DETECTED.value,
                'Face': response,
                'Ignore': None
            }), 404
    elif not face_store_exists:
        return jsonify({
                'status': 'error',
                'Code': RunningCode.FACE_STORE_NOT_EXISTS.value,
                'Face': response,
                'Ignore': None
            }), 404
    else:
        return jsonify({
                'status': 'success',
                'Code': RunningCode.SUCCESS.value,
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
    parser.add_argument('--detection_model', type=str, default='mobilenet',
                        help='detection model type')
    parser.add_argument('--recognition_model', type=str, default='mobilefacenet',
                        help='recognition model type')
    parser.add_argument('--database', type=str, choices=['es', 'milvus'], default='milvus', 
                        help= 'database type')
    args = parser.parse_args()
    
    if args.detection_model == "mobilenet":
        detection_config = read_yaml("detection/configs/RetinaFace_mobilenet025.yaml")
        detection_config['conf'] = 0.9
        detection_config['val_model'] = "/home/zhangjr/pretrained/RetinaFace_MobileNet025_fixed.ckpt"
    recognition_config = {}
    if args.recognition_model == "mobilefacenet":
        recognition_config["backbone"] = args.recognition_model
        recognition_config["pretrained"] = "/home/zhangjr/pretrained/mobile_casia_ArcFace.ckpt"

    if args.database == 'es':
        face_db = FaceEmbeddingES()
    elif args.database == 'milvus':
        face_db = FaceEmbeddingMilvus()

    init_model()

    app.run(host='0.0.0.0', port=5000, debug=True)
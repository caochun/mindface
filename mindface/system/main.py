import argparse
import cv2
import os
import base64

from detection.custom_infer import infer as detection_infer
from recognition.infer import infer as recognition_infer
from detection.runner import read_yaml
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

def get_embedding(image_path,
             detection_model="mobilenet",
             recognition_model="mobilefacenet"):
    
    if detection_model == "mobilenet":
        detection_config = read_yaml("detection/configs/RetinaFace_mobilenet025.yaml")
        detection_config['image_path'] = image_path
        detection_config['conf'] = 0.9
        detection_config['val_model'] = "/home/fangwy/pretrained/Fix_RetinaFace_MobileNet025.ckpt"
    recognition_config = {}
    if recognition_model == "mobilefacenet":
        recognition_config["backbone"] = recognition_model
        recognition_config["pretrained"] = "/home/fangwy/pretrained/mobile_casia_ArcFace.ckpt"
    
    face_img_paths = detection_infer(detection_config)
    if len(face_img_paths) == 0:
        return None
    embeddings = []
    for face_img_path in face_img_paths:
        image_np = cv2.imread(face_img_path)
        image_np = cv2.resize(image_np, (112, 112))
        image_np = np.transpose(image_np, (2, 0, 1))
        embedding = recognition_infer(image_np,
                                  backbone=recognition_config["backbone"],
                                  pretrained=recognition_config["pretrained"])
        # if os.path.exists(face_img_path):
        #     os.remove(face_img_path)
        embeddings.append(embedding)
    return embeddings


# 加载数据库（dict: name -> embedding）
def load_db(DB_PATH):
    DB_PATH = DB_PATH + ".npy"
    if os.path.exists(DB_PATH):
        return np.load(DB_PATH, allow_pickle=True).item()
    return {}

# 保存数据库
def save_db(face_db, DB_PATH):
    np.save(DB_PATH, face_db)

# 计算余弦相似度
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T).item()

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
    DB_PATH = params['reponame']

    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)

    face_db = load_db(DB_PATH)
    embeddings = get_embedding(temp_path)
    if embeddings is None:
        return jsonify({
            'status': 'error',
            'Code': 1008,
            'Message': '注册失败！人脸质量差'
        }), 404
    face_db[identity] = embeddings[0]
    save_db(face_db, DB_PATH)
    print(f"注册成功：{identity}")
    return jsonify({
        'status': 'success',
        'Code': 0,
        'Message': f'ID: {identity} Register to {DB_PATH}'
    }), 0

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
    print("hello")
    identity = params['id']
    DB_PATH = params['reponame']
    face_db = load_db(DB_PATH)
    print("hello")
    if not face_db:
        return jsonify({
            'status': 'error',
            'Code': 1006,
            'Message': '该人脸库不存在'
        }), 404
    print("hello")
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)
    
    query_embs = get_embedding(temp_path)
    for query_emb in query_embs:
        best_score = -1
        best_match = "Unknown"
        for name, emb in face_db.items():
            if emb is None:
                print(f"Warning: 人脸库 {DB_PATH} 中存在不合格的标识 {identity}")
                continue
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_match == identity:
            print("world")
            return jsonify({
                'status': 'success',
                'Code': 0,
                'Confidence': best_score,
                'Message': 'Success'
            }), 200
    return jsonify({
            'status': 'error',
            'Code': 1007,
            'Message': '未检测到人脸'
        }), 404

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
    DB_PATH = params['reponame']
    face_db = load_db(DB_PATH)
    if not face_db:
        return jsonify({
            'status': 'error',
            'Code': 1006,
            'Message': '该人脸库不存在'
        }), 404
    
    # 保存临时文件用于检测模型
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    save_base64_to_jpg(image,temp_path)
    
    query_embs = get_embedding(temp_path)

    response = []
    for query_emb in query_embs:
        best_score = -1
        best_match = "Unknown"
        for name, emb in face_db.items():
            if emb is None:
                print(f"Warning: 人脸库 {DB_PATH} 中存在不合格的标识 {identity}")
                continue
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_score >= threshold:
            candidate = {}
            candidate['Confidence'] = best_score
            candidate['Id'] = best_match
        res = {}
        res['Candidates'] = []
        res['Candidates'].append(candidate)
        res['CandidatesCount'] = 1
        res['CandidatesMessage'] = ""
        res['Feature'] = str(query_emb)
        location = {}
        location['Confidence'] = ''
        location['Height'] = ''
        location['LocalFace'] = ''
        location['Width'] = ''
        location['X'] = ''
        location['Y'] = ''
        res['Location'] = location
        response.append(res)
    print(response)
    return jsonify({
            'status': 'success',
            'Code': 0,
            'Face': response,
            'Ignore': None
        }), 200
  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

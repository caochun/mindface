import argparse
import cv2
import os

from detection.custom_infer import infer as detection_infer
from recognition.infer import infer as recognition_infer
from detection.runner import read_yaml
import numpy as np

DB_PATH = "test.npy"

def get_embedding(image_path,
             detection_model="mobilenet",
             recognition_model="mobilefacenet"):
    
    if detection_model == "mobilenet":
        detection_config = read_yaml("detection/configs/RetinaFace_mobilenet025.yaml")
        detection_config['image_path'] = image_path
        detection_config['conf'] = 0.8
        detection_config['val_model'] = "../../pretrained/Fix_RetinaFace_MobileNet025.ckpt"
    recognition_config = {}
    if recognition_model == "mobilefacenet":
        recognition_config["backbone"] = recognition_model
        recognition_config["pretrained"] = "../../pretrained/mobile_casia_ArcFace.ckpt"
    
    face_img_path = detection_infer(detection_config)
    if face_img_path is None:
        return None
    
    image_np = cv2.imread(face_img_path)
    image_np = cv2.resize(image_np, (112, 112))
    image_np = np.transpose(image_np, (2, 0, 1))
    embedding = recognition_infer(image_np,
                                  backbone=recognition_config["backbone"],
                                  pretrained=recognition_config["pretrained"])
    if os.path.exists(face_img_path):
        os.remove(face_img_path)
    return embedding


# 加载数据库（dict: name -> embedding）
def load_db():
    if os.path.exists(DB_PATH):
        return np.load(DB_PATH, allow_pickle=True).item()
    return {}

# 保存数据库
def save_db(face_db):
    np.save(DB_PATH, face_db)

# 计算余弦相似度
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T).item()

# 注册人脸
def register_face(image_path, identity):
    face_db = load_db()
    embedding = get_embedding(image_path)
    if embedding is None:
        print("注册失败！")
        return
    face_db[identity] = embedding
    save_db(face_db)
    print(f"注册成功：{identity}")

# 识别人脸
def recognize_face(image_path, threshold=0.5):
    face_db = load_db()
    if not face_db:
        print("数据库为空，请先注册人脸。")
        return "Unknown", 0.0

    query_emb = get_embedding(image_path)

    best_score = -1
    best_match = "Unknown"

    for name, emb in face_db.items():
        if emb is None:
            print(f"Warning: register {name} is not correctly!")
            continue
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_match = name
    if best_score >= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score

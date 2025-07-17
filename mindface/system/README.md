
`register_face(image_path, identity)`用于注册人脸信息，其中`image_path`为注册图片路径，`identity`为图片对应的标识。

`recognize_face(image_path, threshold=0.5)`用于识别人脸，其中`image_path`为待识别图片路径。

用户需要根据自身模型存储位置修改`get_embedding`函数中的信息：
````python
    detection_config = read_yaml("detection模型对应的yaml文件路径")
    detection_config['val_model'] = "detection模型路径"
    recognition_config["pretrained"] = "recognition模型路径"
````
# 快速开始
环境准备：
pip install -r mindface/mindface/system/requirement.txt

程序执行：
cd mindface/mindface
python system/main.py

用户需要根据自身模型存储位置以及使用数据库类型修改`main.py`的信息：
````python
    detection_config = read_yaml("detection模型对应的yaml文件路径")
    detection_config['val_model'] = "detection模型路径"
    recognition_config["pretrained"] = "recognition模型路径"
````
暂时只支持mobilenet与mobilefacenet作为检测与识别模型。

模型下载连接：https://box.nju.edu.cn/d/a6a31ba02142470baabc/
`Fix_RetinaFace_MobileNet025.ckpt`为detection模型，对应的yaml文件为`mindface/mindface/detection/configs/RetinaFace_MobileNet025.yaml`；
`mobile_casia_ArcFace.ckpt`为recognition模型

# 接口样式
目前提供的六种接口：
````bash
image_base64=$(base64 -w 0 data/WiderFace/train/images/6--Funeral/6_Funeral_Funeral_6_417.jpg)
curl -X POST "http://localhost:5000/create" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'

curl -X POST "http://localhost:5000/register" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'

curl -X POST "http://localhost:5000/recognize" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'

curl -X POST "http://localhost:5000/recognizeN" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'

curl -X POST "http://localhost:5000/update" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'

curl -X POST "http://localhost:5000/delete" \
   -H "Content-Type: application/json" \
   -d '{"img": "'"$image_base64"'",
      "params":{"id":"AA","reponame":"test"}}'
````

# 数据库选择
我们提供了两种数据库选择，即ElasticSearch和Milvus，搭建服务器的教程分别位于[ES_Tutorial](./ES_Tutorial.md)和[Milvus_Tutorial](./Milvus_Tutorial.md)，请先根据教程于本机搭建数据库服务器，再在main.py中选择需要使用的数据库类型即可。
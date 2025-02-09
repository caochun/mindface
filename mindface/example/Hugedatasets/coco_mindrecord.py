import time
from pycocotools.coco import COCO
from PIL import Image
import os
import gc
import psutil
data_dir = '/disk1/datasets/my_coco'
ann_file_train = f'{data_dir}/annotations/instances_train2017.json'
img_dir_train = f'{data_dir}/train2017/'
from PIL import Image
from pycocotools.coco import COCO
from mindspore import log as logger
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,ConcatDataset,Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import json
from torchvision.transforms import functional as F


pid = os.getpid()
process = psutil.Process(pid)
from itertools import chain,tee
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="", type=int)
    args = parser.parse_args()
    return args

def memory_log():
    print(process.memory_info().rss / 1024 / 1024,"MB")
    return process.memory_info().rss / 1024 / 1024


class coco_DataSet:
                                             
    def __init__(self, coco_root, transforms, train_set=True, pieces=1):
        self.transforms = transforms
        self.pieces = pieces  
        self.annotations_root = os.path.join(coco_root, "annotations")
        self.annotations_json = os.path.join(self.annotations_root, "instances_train2017.json")
        self.image_root = os.path.join(coco_root, "train2017")
        assert os.path.exists(self.annotations_json), "{} file not exist.".format(self.annotations_json)
        json_file = open(self.annotations_json, 'r')
        self.coco_dict = json.load(json_file)
        self.bbox_image = {}
        for temp in self.coco_dict["annotations"]:
            temp_append = []
            pic_id = temp["image_id"] - 1
            bbox = temp["bbox"]
            class_id = temp["category_id"]
            temp_append.append(class_id)
            temp_append.append(bbox)
            if self.bbox_image.__contains__(pic_id):
                self.bbox_image[pic_id].append(temp_append)
            else:
                self.bbox_image[pic_id] = []
                self.bbox_image[pic_id].append(temp_append)
            
        self.image_list = self.coco_dict["images"]
        self.image_list_now = self.image_list[:100000] * self.pieces

    def __len__(self):
        return len(self.image_list_now)

    def __getitem__(self, idx):
        image_info = self.image_list_now[idx]
        pic_name = image_info["file_name"]
        pic_path = os.path.join(self.image_root, pic_name)

        image = Image.open(pic_path).convert("RGB")
        image = image.resize((256, 256))
        image = np.array(image)
        bboxes = []
        labels = []
        target = {}

        pic_id = image_info["id"] - 1  # 根据 JSON 文件的图像索引
        if self.bbox_image.__contains__(pic_id):
            for annotations in self.bbox_image[pic_id]:
                bboxes.append(annotations[1])
                labels.append(annotations[0])
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target["boxes"] = bboxes
            target["labels"] = labels
        else:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target["boxes"] = bboxes
            target["labels"] = labels
        if (self.transforms) :
            self.transforms = transforms.Resize((256, 256))
            image = self.transforms(image)
        return image,target



class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image):
        image = F.to_tensor(image)
        return image

batch_size_for_all = 32


def pytorch_memory(pieces):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    batch_size = batch_size_for_all
    
    coco_custom = coco_DataSet("/disk1/datasets/my_coco", transforms, True, pieces)
    train_loader = DataLoader(dataset=coco_custom ,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

    print("The number of actual samples: {}".format(100000*pieces)) 

    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in train_loader:
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == samples:
            memory = memory_log()

            break
    end = time.time()
    cost_time = end - start
    print ("Lafa reading time: {}".format(cost_time))
    # for i in throughput_list:
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))

    import csv
    output_file = "/home/coco/coco_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])


    return memory,sum(throughput_list)/len(throughput_list), cost_time




def tensorflow_memory(pieces):
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    from pycocotools.coco import COCO

    batch_size = batch_size_for_all

    class COCO_Dataset_tensorflow:
        def __init__(self, coco_root, image_size=(256, 256), train_set=True, pieces=1):
            self.pieces = pieces  # 数据集重复倍数
            self.image_size = image_size  # 图像的目标尺寸

            # 数据集路径
            self.annotations_root = os.path.join(coco_root, "annotations")
            self.annotations_json = os.path.join(self.annotations_root, "instances_train2017.json")
            self.image_root = os.path.join(coco_root, "train2017")

            assert os.path.exists(self.annotations_json), f"{self.annotations_json} file not exist."

            # 读取 COCO JSON 数据
            with open(self.annotations_json, 'r') as json_file:
                self.coco_dict = json.load(json_file)

            self.bbox_image = {}
            for temp in self.coco_dict["annotations"]:
                temp_append = []
                pic_id = temp["image_id"] - 1
                bbox = temp["bbox"]
                class_id = temp["category_id"]
                temp_append.append(class_id)
                temp_append.append(bbox)
                if pic_id in self.bbox_image:
                    self.bbox_image[pic_id].append(temp_append)
                else:
                    self.bbox_image[pic_id] = [temp_append]

            self.image_list = self.coco_dict["images"]
            self.image_list = self.image_list[:100000] * self.pieces

        def __len__(self):
            return len(self.image_list)

        def parse_image(self, idx):
            image_info = self.image_list[idx]
            pic_name = image_info["file_name"]
            pic_path = os.path.join(self.image_root, pic_name)

            # 加载并预处理图像
            image = Image.open(pic_path).convert("RGB")
            image = image.resize(self.image_size)
            image = np.array(image, dtype=np.float32) / 255.0  # 归一化到 [0,1]

            # 提取目标框和标签
            bboxes = []
            labels = []

            pic_id = image_info["id"] - 1
            if pic_id in self.bbox_image:
                for annotation in self.bbox_image[pic_id]:
                    labels.append(annotation[0])
                    bboxes.append(annotation[1])

            bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

            return image, {"boxes": bboxes, "labels": labels}

        def generator(self):
            for idx in range(len(self)):
                yield self.parse_image(idx)

        def to_tf_dataset(self, batch_size=batch_size, shuffle=False, num_parallel_calls=8):
            dataset = tf.data.Dataset.from_generator(
                self.generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
                    {
                        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                        "labels": tf.TensorSpec(shape=(None,), dtype=tf.int64),
                    },
                ),
            )

            if shuffle:
                dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset



    coco_dataset = COCO_Dataset_tensorflow(coco_root="/disk1/datasets/my_coco", image_size=(256, 256), pieces=pieces)

    train_dataset = coco_dataset.to_tf_dataset(batch_size=batch_size, shuffle=False, num_parallel_calls=8)
    print("The number of actual samples: {}".format(100000*pieces)) 

    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in train_dataset:
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == samples:
            memory = memory_log()

            break
    end = time.time()
    cost_time = end - start
    print ("Lafa reading time: {}".format(cost_time))

    import csv
    output_file = "/home/coco/coco_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time





def mindspore_memory(pieces):
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.transforms as transforms


    transform = transforms.Compose([
        vision.Resize((256, 256)),
        ToTensor()
    ])
    batch_size = batch_size_for_all
    paralled_workers = 8

    coco_custom = coco_DataSet("/disk1/datasets/my_coco", transforms=transform, train_set=True, pieces=pieces)
    dataset = ds.GeneratorDataset(source=coco_custom, column_names=["image"], num_parallel_workers=8)
    dataset = dataset.batch(batch_size, num_parallel_workers=paralled_workers)
    # dataset.save('/data0/hugedataset_mr/coco100000.Lafa')

    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))


    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in dataset.create_tuple_iterator(output_numpy=True):
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == samples:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Lafa reading time: {}".format(cost_time))
    # for i in throughput_list:
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))


    import csv
    output_file = "/home/coco/coco_mindspore.csv"
    mode = 'w' if samples == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])


    return memory,sum(throughput_list)/len(throughput_list), cost_time



def Lafa_memory(pieces):
    import LafaDataset
    dataset = LafaDataset(["/home/coco.lafa"]*pieces, shuffle=True, num_parallel_workers=8)

    def decode_image(file_name, img_id):
        try:
            image_root = "/data1/wangcong/coco2017/train2017/"
            if isinstance(file_name, np.ndarray):
                file_name = file_name.tobytes().decode("utf-8").strip()  # 确保字符串解码正确
            else:
                raise TypeError(f"Unexpected file_name type: {type(file_name)}.")
                if "\x00" in file_name:
                    raise ValueError(f"File name contains null byte: {file_name}")
            # 构建完整的文件路径
            image_path = os.path.join(image_root, file_name)
            print(f"Decoded file name: {image_path}")

            # 尝试读取图片
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

            return image, img_id  # 返回解码后的图片和对应的 ID

        except Exception as e:
            # 捕获异常并记录
            print(f"Error processing file_name: {file_name}, img_id: {img_id}, Error: {e}")
            raise e  # 重新抛出异常以终止处理

    dataset = dataset.map(operations=decode_image, input_columns=["file_name", "id"],
                        output_columns=["image", "id"]
                        )

    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))

    batch_size = batch_size_for_all
    paralled_workers = 8
    dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)


    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in dataset.create_tuple_iterator(output_numpy=True):
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == samples:
            memory = memory_log()

            break
    end = time.time()
    cost_time = end - start
    print ("Lafa reading time: {}".format(cost_time))
    import csv
    output_file = "/home/coco/coco_Lafa.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
    return memory,sum(throughput_list)/len(throughput_list), cost_time




def Lafa():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]:#1000w-5e
        memory,throughput,cost_time = Lafa_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("Lafa:\n")
    print('Memory:')
    for i in range(len(memory_list)):
        print(memory_list[i])
    print("Throughput:")
    for i in range(len(memory_list)):
        print(throughput_List[i])
    print("Cost time:")
    for i in range(len(memory_list)):
        print(time_list[i])
    return 0




if __name__ == "__main__":
    args = parse_args()
    Lafa_memory(args.pieces)


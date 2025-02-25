import os
import time
import csv
# import torch
# import torchvision
import psutil
# from torch.utils.data import DataLoader, Dataset
import mindspore.dataset.audio as audio
import glob
import xml.etree.ElementTree as ET
import os
import time
from PIL import Image
import numpy as np
import math
# import torchaudio
import gc
import csv
pid = os.getpid()
process = psutil.Process(pid)
import json
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
# import librosa
import mindspore as ms

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="", type=int)
    args = parser.parse_args()
    return args



from mindspore import log as logger
def memory_log():
    print(process.memory_info().rss / 1024 / 1024,"MB")
    return process.memory_info().rss / 1024 / 1024
from mindspore.mindrecord import FileWriter



class ShapeNetDataset:
    
    def __init__(self, root_path, split='train', num_points=1024,pieces = 1, normal_channel=False, class_choice=None):
        self.path = root_path
        self.num_points = num_points
        self.normal_channel = normal_channel
        self.pieces = pieces
        self.catfile = os.path.join(self.path, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.path, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Existing..' % split)
                exit(1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}
        self.cache_size = 0

        self.datapath = self.datapath[:10000] * 100


        schema = {
            "category": {"type": "string"},  # 存储类别名称
            "file_path": {"type": "string"}  # 存储文件路径
        }

        # 初始化 MindRecord Writer
        writer = FileWriter(file_name="/data1/wangke_indexonly/ShapeNet.mindrecord", shard_num=1)

        # 设置 schema
        writer.add_schema(schema, "ShapeNet Dataset Schema")

        # 准备数据
        data = []
        for item, fn in self.datapath:
            record = {
                "category": item,
                "file_path": fn
            }
            data.append(record)

        # 写入数据到 MindRecord
        writer.write_raw_data(data)
        writer.commit()
        print(f"datapath successfully saved!!!!!")




    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            # if len(self.cache) < self.cache_size:
            #     self.cache[index] = (point_set, cls, seg)
        #point_set[:, 0:3] = self._pc_normalize(point_set[:, 0:3])

        if len(seg) >= self.num_points:
            point_set = point_set[:self.num_points, :]
            seg = seg[:self.num_points]
        else:
            numbers = self.num_points - len(seg)
            zeros = np.array(point_set[0, :]).reshape(1, -1)
            zeros = zeros.repeat(numbers, axis=0)
            point_set = point_set[:self.num_points, :]
            seg_zeros = np.array(seg[0]).repeat(numbers, axis=0)
            seg = seg[:self.num_points]
            point_set = np.concatenate((point_set, zeros), axis=0)
            seg = np.concatenate((seg, seg_zeros), axis=0)

        return point_set#, cls, seg

    def __len__(self):
        return len(self.datapath)


def pytorch_memory(pieces):
    ShapeNet_dataset_dir = "/disk1/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    
    # Load dataset
    dataset = ShapeNetDataset(root_path=ShapeNet_dataset_dir, 
                                            num_points = 1024,
                                            pieces = pieces,
                                            split='train', 
                                            normal_channel= False)

  
    dataloader = DataLoader(dataset, num_workers=8)# batch_size=batch_size
   
    batch_size = 32
    # dataset = dataset.batch(batch_size, num_parallel_workers=num_parallel_workers)
    


    # samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(10000*pieces)) 
    # 测试数据加载和处理速度

    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in dataloader:
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == 100:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindspore reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(10000*pieces,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ShapeNet40/ShapeNet_pytorch.csv"
    # 'a' = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time






def tensorflow_memory(pieces):
    import time
    import numpy as np
    import tensorflow as tf
    
    """Load 'a'lNet40 dataset in TensorFlow."""
    ShapeNet_dataset_dir = "/disk1/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    dataset_generator = ShapeNetDataset(root_path=ShapeNet_dataset_dir, 
                                            num_points = 1024,
                                            pieces = pieces,
                                            split='train', 
                                            normal_channel= False)

    output_signature = (
        tf.TensorSpec(shape=(1024, 6 if dataset_generator.normal_channel else 3), dtype=tf.float32))
    
    dataset = tf.data.Dataset.from_generator(lambda: dataset_generator, output_signature=output_signature)

    # # Apply transformations for training
    # dataset = dataset.map(lambda point_set, cls, seg: (augment_data(point_set), cls, seg), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # # Batch the dataset
    # dataset = dataset.batch(batch_size, drop_remainder=True)

    batch_size = 32
    # dataset = dataset.batch(batch_size, num_parallel_workers=num_parallel_workers)
    


    # samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(10000*pieces)) 
    # 测试数据加载和处理速度

    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in dataset:
        data_nums += 1
        if data_nums % batch_size == 0:
            throughput_end = time.time()
            throughput_time = throughput_end - throughput_start
            throughput = batch_size / throughput_time
            throughput_list.append(throughput) 
            throughput_start = time.time()

        if data_nums == 100:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindspore reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(10000*pieces,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ShapeNet40/ShapeNet_tensorflow.csv"
    # 'a' = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time


    return memory,sum(throughput_list)/len(throughput_list), cost_time



def mindspore_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    data_nums=0

    ShapeNet_dataset_dir = "/disk1/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    #dataset = ds.KITTIDataset(dataset_dir=kitti_dataset_dir, usage="train",decode=False,num_parallel_workers=num_parallel_workers)
    #dataset = ds.ImageFolderDataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_parallel_workers, decode=False)

    dataset_generator = ShapeNetDataset(root_path=ShapeNet_dataset_dir, 
                                            num_points = 1024,
                                            pieces = pieces,
                                            split='train', 
                                            normal_channel= False)

    dataset = ds.GeneratorDataset(dataset_generator, ["point_set"],#
                                    shuffle=False, num_parallel_workers = 8
                                    )
    # dataset.save("/data1/ShapeNet_10000.mindrecord")
    


    # # trans = [random_scale_point_cloud_class(), shift_point_cloud_class()]
    # dataset = dataset.map(operations=trans,
    #                     input_columns="point_set",
    #                     num_parallel_workers=num_parallel_workers)
    # dataset = dataset.batch(batch_size=10, drop_remainder=True)


    batch_size = 32
    # dataset = dataset.batch(batch_size, num_parallel_workers=num_parallel_workers)
    
    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples)) 
    # 测试数据加载和处理速度

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

        if data_nums == 100:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindspore reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ShapeNet40/ShapeNet_mindspore.csv"
    # 'a' = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time


    return memory,sum(throughput_list)/len(throughput_list), cost_time


def mindrecord_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    data_nums=0

    dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/ShapeNet.mindrecord"] * pieces, shuffle=False)



    batch_size = 32
    # dataset = dataset.batch(batch_size, num_parallel_workers=num_parallel_workers)
    
    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples)) 
    # 测试数据加载和处理速度

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

        if data_nums == 1000:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindspore reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ShapeNet40/ShapeNet_indexonly.csv"
    # 'a' = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time


    return memory,sum(throughput_list)/len(throughput_list), cost_time






def mindrecord():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,20000,25000,30000,4000,50000]:#1000w-5e
        memory,throughput,cost_time = mindrecord_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("mindrecord:\n")
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



def mindspore():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,20000,25000,30000,4000,50000]:#1000w-5e
        memory,throughput,cost_time = mindspore_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("mindspore:\n")
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


def tensorflow():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,20000,25000,30000,4000,50000]:#1000w-5e
        memory,throughput,cost_time = tensorflow_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("tensorflow:\n")
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

def pytorch():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,20000,25000,30000,40000,50000]:#1000w-5e
        memory,throughput,cost_time = pytorch_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("pytorch:\n")
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
    mindrecord_memory(args.pieces)
    # mindspore_memory(1)

    # mindspore_memory(args.pieces)
    # pytorch_memory(args.pieces)
    # tensorflow_memory(args.pieces)
    # mindrecord_memory(11000)
    # mindspore_memory(2000)
    # mindspore_memory(3000)
    # mindspore_memory(4)
    # pytorch_memory(1000)
    # tensorflow_memory(1000)
    # mindspore() 
    # mindrecord()
    # pytorch()
    # tensorflow()
    # mindrecord()
    # tensorflow()
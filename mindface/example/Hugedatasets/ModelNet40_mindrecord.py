import os
import time
import numpy as np
# import torch
# import torchvision
import mindspore.dataset as ds
import mindspore.dataset.vision as C
# from torch.utils.data import Dataset,DistributedSampler
import tensorflow as tf
# from torchvision import transforms
import psutil
import csv
import gc
pid = os.getpid()
process = psutil.Process(pid)
from mindspore import log as logger
def memory_log():
    print(process.memory_info().rss / 1024 / 1024,"MB")
    return process.memory_info().rss / 1024 / 1024
from mindspore.mindrecord import FileWriter
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="", type=int)
    args = parser.parse_args()
    return args


batch_size_for_all = 1 #32
num_parallel_workers = 8
epoch = 1
class ModelNet40Dataset:
    
    def __init__(self, root_path, split, use_norm, pieces):
        """read datasets path"""
        self.path = root_path
        self.split = split
        self.use_norm = use_norm
        self.pieces = pieces
        self.datapath_now = []
        shapeid_path = "modelnet40_train.txt" if self.split == "train" else "modelnet40_test.txt"
        catfile = os.path.join(self.path, "modelnet40_shape_names.txt")
        cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(cat, range(len(cat))))
        shape_ids = [line.rstrip() for line in open(os.path.join(self.path, shapeid_path))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        self.datapath = [(shape_names[i], os.path.join(self.path, shape_names[i], shape_ids[i]) + '.txt') for i
                    in range(len(shape_ids))]
        # print(f"Length of datapath: {len(self.datapath)}")
        self.datapath_now =(self.datapath[:5000]) * 200



        schema = {
                    "shape_name": {"type": "string"},  # 存储形状名称
                    "file_path": {"type": "string"}   # 存储文件路径
                }

        # 初始化 MindRecord Writer
        writer = FileWriter(file_name="/data1/wangke_indexonly/ModelNet40.mindrecord", shard_num=1, overwrite=True)

        # 设置 schema
        writer.add_schema(schema, "ModelNet40 Dataset Schema")

        # 准备数据
        data = []
        for shape_name, file_path in self.datapath_now:
            record = {
                "shape_name": shape_name,
                "file_path": file_path
            }
            data.append(record)

        # 写入数据到 MindRecord
        writer.write_raw_data(data)
        writer.commit()
        print(f"datapath_now successfully saved to!")




    def __getitem__(self, index):
        """get item"""
        fn = self.datapath_now[index]
        label = self.classes[self.datapath_now[index][0]]
        label = np.asarray([label]).astype(np.int32)
        point_cloud = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        if self.use_norm:
            point_cloud = point_cloud[:, :]
        else:
            point_cloud = point_cloud[:, :3]
        return point_cloud, label[0]
    def translate_pointcloud(self,pointcloud):
        """translate"""
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud
    def __len__(self):
        """len"""
        return len(self.datapath_now)


def pytorch_memory(pieces):
    
    batch_size = batch_size_for_all
    paralled_workers = 8

    ModelNet40_dataset_dir = "/disk1/datasets/modelnet40_normal_resampled"
    dataset_generator = ModelNet40Dataset(root_path = ModelNet40_dataset_dir,
                                        split = 'train',
                                        use_norm = False,
                                        pieces = pieces)
    dataloader = torch.utils.data.DataLoader(dataset_generator, batch_size=batch_size, shuffle=False, num_workers=8)
    print("The number of actual samples: {}".format(5000*pieces)) 
    batch_size = batch_size_for_all
    paralled_workers = 8
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

        if data_nums == 1000:
            memory = memory_log()
        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(5000*pieces,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ModelNet40/ModelNet40_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])


    return memory,sum(throughput_list)/len(throughput_list), cost_time



def tensorflow_memory(pieces):
    """Load ModelNet40 dataset in TensorFlow."""
    ModelNet40_dataset_dir = "/disk1/datasets/modelnet40_normal_resampled"
    dataset_generator = ModelNet40Dataset(root_path = ModelNet40_dataset_dir,
                                         split = 'train',
                                         use_norm = False,
                                         pieces = pieces)
    # Create a tf.data.Dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    batch_size = batch_size_for_all
    paralled_workers = 8
    dataset = dataset.batch(batch_size)
    print("The number of actual samples: {}".format(5000*pieces)) 
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

        if data_nums == 1000:
            memory = memory_log()

        # if data_nums == 2000:
        #     memory_log("nexting data")
            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(5000*pieces,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ModelNet40/ModelNet40_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time



def mindspore_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision

    ModelNet40_dataset_dir = "/disk1/datasets/modelnet40_normal_resampled"
    dataset_generator = ModelNet40Dataset(root_path = ModelNet40_dataset_dir,split = 'train',use_norm = False, pieces = pieces)
    dataset = ds.GeneratorDataset(source = dataset_generator, column_names=["data", "label"], shuffle=False,num_parallel_workers=num_parallel_workers)
    batch_size = batch_size_for_all
    paralled_workers = 8


    # dataset.save('/data1/modelnet40_5000.mindrecord',8)
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)

    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))# 117266  
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
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ModelNet40/ModelNet40_mindspore.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time




def mindrecord_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision

    dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/ModelNet40.mindrecord"]*pieces, shuffle=True, num_parallel_workers=8)


    batch_size = batch_size_for_all
    paralled_workers = 8
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)
    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))# 117266  
    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in enumerate(dataset.create_tuple_iterator(output_numpy=True)):
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
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/ModelNet40/ModelNet40_indexonly.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time

def mindrecord():
    memory_list = []
    throughput_List = []
    time_list = []
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#
        memory,throughput,cost_time = mindrecord_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("mindrecord:")
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
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#1000w-5e
        memory,throughput,cost_time = mindspore_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("Mindspore:")
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
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#1000w-5e
        memory,throughput,cost_time = tensorflow_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("Tensorflow:")
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
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#1000w-5e
        memory,throughput,cost_time = pytorch_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("Pytorch:")
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
    # mindrecord()
    # mindspore()
    # tensorflow()
    # pytorch()
    # mindspore()
    # mindspore_memory(1)
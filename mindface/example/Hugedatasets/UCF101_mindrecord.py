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


import mindspore
from mindspore import dataset as ds
from mindspore.dataset import transforms
from mindspore.dataset.transforms import c_transforms
from mindspore import dtype as mstype
import mindspore.dataset.vision as vision
import mindspore as ms
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size

batch_size_for_all = 1
from mindspore.mindrecord import FileWriter
class VideoDataset:
        def __init__(self, pieces):
            self.frames_per_clip = 16
            self.step_between_clips = 10
            data_num=0
            txt_file_path = '/disk1/datasets/ucfTrainTestlist/testlist01.txt'
            self.video_paths = []
            
            with open(txt_file_path, 'r') as f:
                for line in f:
                    # 假设 testlist01.txt 的每行是：视频路径 视频类别
                    video_info = line.strip().split()
                    video_path = video_info[0]  # 视频路径
                    result = "".join(['/disk1/datasets/UCF101/', video_path])
                    
                    self.video_paths.append(result)

            self.video_paths = self.video_paths[:1000] * 1000



            schema = {
                "video_path": {"type": "string"}  # 存储视频文件路径
            }

            # 初始化 MindRecord Writer
            writer = FileWriter(file_name="/data1/wangke_indexonly/UCF101.mindrecord", shard_num=1)

            # 设置 schema
            writer.add_schema(schema, "VideoDataset Schema")

            # 准备数据
            data = []
            for video_path in self.video_paths:
                record = {"video_path": video_path}
                data.append(record)

            # 写入数据到 MindRecord
            writer.write_raw_data(data)
            writer.commit()
            print(f"Video paths successfully saved!")










        def __getitem__(self, index):

            video_clips = []
            video_path = self.video_paths[index]

            video_output, _ , _ = vision.read_video(video_path)


            for start_frame in range(0, len(video_output) - self.frames_per_clip + 1, self.step_between_clips):
                clip = video_output[start_frame:start_frame + self.frames_per_clip]
                video_clips.append(clip)

                #滑动窗口

            return video_clips[0]

        def __len__(self):
            # 返回数据集的长度
            return len(self.video_paths)
        def generator(self):
            """Generator function for tf.data.Dataset."""
            for i in range(len(self)):
                yield self.__getitem__(i)


def pytorch_memory(pieces):

    
    # 设置目标视频尺寸
    target_video_height = 240  # 目标高度
    target_video_width = 320  # 目标宽度

    batch_size = batch_size_for_all
    paralled_workers = 8

    
    dataset = VideoDataset(pieces)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)


    # dataset = dataset.map(operations=[ds.vision.Resize((target_video_height, target_video_width))], input_columns=["video_clip"], num_parallel_workers=8)

    # dataset = dataset.batch(batch_size,num_parallel_workers=dataset_num_workers, drop_remainder=True)


    samples = 1000 * pieces
    print("The number of actual samples: {}".format(samples))# 117266  


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
        if data_nums == 10:
            memory = memory_log()

            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))


    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/UCF101/UCF_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time



def tensorflow_memory(pieces):
    import cv2
    import numpy as np
    import time
    import mindspore
    from mindspore import dataset as ds
    from mindspore.dataset import transforms
    from mindspore.dataset.transforms import c_transforms
    from mindspore import dtype as mstype
    import mindspore.dataset.vision as vision
    import mindspore as ms
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    
    # 设置目标视频尺寸
    target_video_height = 240  # 目标高度
    target_video_width = 320  # 目标宽度
    
    dataset = VideoDataset(pieces)

        # 使用 GeneratorDataset 创建数据集
    dataset = tf.data.Dataset.from_generator(
        dataset.generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32)
        )
    )

    # dataset = dataset.map(operations=[ds.vision.Resize((target_video_height, target_video_width))], input_columns=["video_clip"], num_parallel_workers=8)

    # dataset = dataset.batch(batch_size,num_parallel_workers=dataset_num_workers, drop_remainder=True)

    batch_size = batch_size_for_all
    paralled_workers = 8

    samples = pieces * 1000
    print("The number of actual samples: {}".format(samples))# 117266  


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
        if data_nums == 10:
            memory = memory_log()


            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/UCF101/UCF_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time









def mindspore_memory(pieces):
    import cv2
    import numpy as np
    import time
    import mindspore
    from mindspore import dataset as ds
    from mindspore.dataset import transforms
    from mindspore.dataset.transforms import c_transforms
    from mindspore import dtype as mstype
    import mindspore.dataset.vision as vision
    import mindspore as ms
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    
    # 设置目标视频尺寸
    target_video_height = 240  # 目标高度
    target_video_width = 320  # 目标宽度
    
    dataset = VideoDataset(pieces)

        # 使用 GeneratorDataset 创建数据集
    dataset = ds.GeneratorDataset(dataset, column_names=["video_clip"])

    # dataset = dataset.map(operations=[ds.vision.Resize((target_video_height, target_video_width))], input_columns=["video_clip"], num_parallel_workers=8)

    # dataset = dataset.batch(batch_size,num_parallel_workers=dataset_num_workers, drop_remainder=True)

    batch_size = batch_size_for_all
    paralled_workers = 8

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
        if data_nums == 10:
            memory = memory_log()


            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/UCF101/UCF_mindspore.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time

def mindrecord_memory(pieces):
    import cv2
    import numpy as np
    import time
    import mindspore
    from mindspore import dataset as ds
    from mindspore.dataset import transforms
    from mindspore.dataset.transforms import c_transforms
    from mindspore import dtype as mstype
    import mindspore.dataset.vision as vision
    import mindspore as ms
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    
    # 设置目标视频尺寸
    target_video_height = 240  # 目标高度
    target_video_width = 320  # 目标宽度
    
    dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/UCF101.mindrecord"]*pieces, shuffle=True, num_parallel_workers=8)


    # dataset = dataset.map(operations=[ds.vision.Resize((target_video_height, target_video_width))], input_columns=["video_clip"], num_parallel_workers=8)

    # dataset = dataset.batch(batch_size,num_parallel_workers=dataset_num_workers, drop_remainder=True)

    batch_size = batch_size_for_all
    paralled_workers = 8

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


            break
    end = time.time()
    cost_time = end - start
    print ("Mindrecord reading time: {}".format(cost_time))
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/UCF101/UCF_indexonly.csv"
    # mode = 'w' if pieces == 100 else 'a'
    with open(output_file, 'a') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time

    




    

if __name__ == "__main__":
    args = parse_args()
    mindrecord_memory(args.pieces)
    # mindspore_memory(1)
    # pytorch_memory(20000)
    # tensorflow_memory(1)
    # mindspore_memory(args.pieces)
    # pytorch_memory(args.pieces)
    # tensorflow_memory(args.pieces)
    # mindrecord_memory(1)
    # mindspore()
    # tensorflow()
    # pytorch()
    # mindspore()
    # mindspore_memory(1)
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
from mindspore import log as logger
def memory_log():
    print(process.memory_info().rss / 1024 / 1024,"MB")
    return process.memory_info().rss / 1024 / 1024

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="", type=int)
    args = parser.parse_args()
    return args



import librosa
#2426


batch_size_for_all = 1


def pytorch_memory(pieces):
    n_fft = 1024
    win_length = None
    hop_length = 512

    batch_size = batch_size_for_all
    num_workers = 8

    class LJSpeechDataset(Dataset):
        def __init__(self, data_dir, pieces=1):
            """
            LJSpeech 数据集初始化。
            :param data_dir: 数据集根目录
            :param pieces: 数据集重复倍数
            """
            self.audio_paths = self._load_audio_paths(data_dir, pieces)

        def _load_audio_paths(self, data_dir, pieces):
            # 读取 metadata.csv 文件
            metadata_file = os.path.join(data_dir, 'metadata.csv')
            assert os.path.exists(metadata_file), f"{metadata_file} 文件不存在。"

            with open(metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 获取音频路径列表
            audio_paths = []
            for line in lines:
                parts = line.strip().split('|')
                audio_paths.append(os.path.join(data_dir, 'wavs', f"{parts[0]}.wav"))

            # 截取前 2000 条数据，并进行重复
            audio_paths = audio_paths[:2000] * pieces
            return audio_paths

        def __getitem__(self, index):
            # 加载音频数据
            audio_path = self.audio_paths[index]
            audio, sr = librosa.load(audio_path, sr=22050)  # 使用 librosa 加载音频
            return torch.tensor(audio, dtype=torch.float32)

        def __len__(self):
            return len(self.audio_paths)

    # 数据路径
    data_dir = '/disk1/datasets/LJSpeech-1.1'

    # 创建数据集实例
    dataset = LJSpeechDataset(data_dir, pieces=pieces)

    # 使用 PyTorch DataLoader
    data_loader = DataLoader(dataset, num_workers=num_workers, shuffle=False) #

    print("The number of actual samples: {}".format(2000*pieces)) 
    # 测试数据加载和处理速度

    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in data_loader:
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
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))\


    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/LJS/LJS_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])


    return memory,sum(throughput_list)/len(throughput_list), cost_time



def tensorflow_memory(pieces):
    import time
    import numpy as np
    import tensorflow as tf
    
    import librosa
    print("Eager execution enabled:", tf.executing_eagerly())


    n_fft = 1024
    win_length = None
    hop_length = 512

    batch_size = batch_size_for_all
    paralled_workers = 8

    # class LJSpeechDataset(tf.data.Dataset):
    #     def __new__(cls, data_dir,pieces):
    #         # 读取文件名和路径
            

    #         metadata_file = os.path.join(data_dir, 'metadata.csv')
    #         with open(metadata_file, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
            
    #         # 解析元数据，获取音频路径和文本信息
    #         audio_paths = []
            
    #         for line in lines:
    #             parts = line.strip().split('|')
    #             audio_paths.append(os.path.join(data_dir, 'wavs', f"{parts[0]}.wav"))
                
    #         audio_paths_now = audio_paths[:2000] * pieces
    #         # 构造数据集
    #         dataset = tf.data.Dataset.from_tensor_slices(audio_paths_now)
            
    #         return dataset
    class LJSpeechDataset:
        def __init__(self, data_dir, pieces=1):

            self.audio_paths = self._load_audio_paths(data_dir, pieces)

        def _load_audio_paths(self, data_dir, pieces):
            # 读取 metadata.csv 文件
            metadata_file = os.path.join(data_dir, 'metadata.csv')
            assert os.path.exists(metadata_file), f"{metadata_file} 文件不存在。"

            with open(metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 获取音频路径列表
            audio_paths = []
            for line in lines:
                parts = line.strip().split('|')
                audio_paths.append(os.path.join(data_dir, 'wavs', f"{parts[0]}.wav"))

            # 截取前 2000 条数据，并进行重复
            audio_paths = audio_paths[:2000] * pieces
            return audio_paths

        def __getitem__(self, index):
            # 加载音频数据
            audio_path = self.audio_paths[index]
            audio, sr = librosa.load(audio_path, sr=22050)  # 使用 librosa 加载音频

            return audio
            # return np.array(audio, dtype=np.float32)

        def __len__(self):
            return len(self.audio_paths)

    # 设置数据路径和参数
    data_dir = '/disk1/datasets/LJSpeech-1.1'  # LJSpeech 数据集路径
    
    dataset = LJSpeechDataset(data_dir, pieces=pieces)

    # 使用 PyTorch DataLoader
    # data_loader = DataLoader(dataset, num_workers=num_workers, shuffle=False) #



    dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # def preprocess_data(audio_path):
    #     # 使用 librosa 加载音频文件
    #     audio_path = audio_path.numpy().decode('utf-8')
    #     audio, sr = librosa.load(audio_path, sr=22050)  # 默认采样率 22050Hz
    #     # 提取梅尔频谱
    #     return audio



    # dataset = LJSpeechDataset(data_dir,pieces)


    # dataset = dataset.map(lambda x: tf.py_function(preprocess_data, [x], [tf.float32]), 
    #                         num_parallel_calls=paralled_workers)
    
        # 数据集批处理，并设置填充规则
    #dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, 128]),))

    dataset = dataset.batch(batch_size)
    # 数据集的预取
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("The number of actual samples: {}".format(2000*pieces)) 

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
    print("Average throughput of batch at sample number {} is {} items/sec".format(samples,sum(throughput_list)/len(throughput_list)))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/shuffle/LJS/LJS_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    return memory,sum(throughput_list)/len(throughput_list), cost_time






def mindspore_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import librosa
    import mindspore as ms


    n_fft = 1024
    win_length = None
    hop_length = 512


    # 自定义数据集类
    class LJSpeechDataset:
        def __init__(self, data_dir, pieces=1):
            from mindspore.mindrecord import FileWriter
            self.audio_paths = self._load_audio_paths(data_dir, pieces)

            self.audio_paths = self.audio_paths * 50

            schema = {
                "audio_path": {"type": "string"}  # 存储音频路径
            }

            writer = FileWriter(file_name="/data1/wangke_indexonly/LJSpeech_short_10w.mindrecord", shard_num=1)
            writer.add_schema(schema, "LJSpeech Dataset Schema")

            # 准备数据
            data = [{"audio_path": path} for path in self.audio_paths]

            # 写入数据到 MindRecord
            writer.write_raw_data(data)
            writer.commit()

            print(f"Audio paths successfully saved!")




        def _load_audio_paths(self, data_dir, pieces):
            # 读取 metadata.csv 文件
            metadata_file = os.path.join(data_dir, 'metadata.csv')
            assert os.path.exists(metadata_file), f"{metadata_file} 文件不存在。"

            with open(metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 获取音频路径列表
            audio_paths = []
            for line in lines:
                parts = line.strip().split('|')
                audio_paths.append((f"{parts[0]}.wav"))

            # 截取前 2000 条数据，并进行重复
            audio_paths = audio_paths[:2000] * pieces
            return audio_paths



        def __getitem__(self, index):
            # 加载音频数据
            audio_path = self.audio_paths[index]
            audio, sr = librosa.load(audio_path, sr=22050)  # 使用 librosa 加载音频

            return audio
            # return np.array(audio, dtype=np.float32)

        def __len__(self):
            return len(self.audio_paths)

    # 数据路径
    data_dir = '/disk1/datasets/LJSpeech-1.1'

    # 创建数据集实例
    dataset = LJSpeechDataset(data_dir, pieces)


    

    batch_size = batch_size_for_all
    paralled_workers = 8


    # MindSpore GeneratorDataset
    dataset = ds.GeneratorDataset(
        source=dataset,
        column_names=["audio"],
        num_parallel_workers=paralled_workers
    )

    # dataset.save('/data1/LJS_sr1000_2000.mindrecord')
   
    dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)

    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))# 117266  
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)

    #     # 定义归一化函数：处理 waveform
    # def normalize_waveform(waveform):
    #     """
    #     将 waveform 转换为一维并归一化到目标长度。
    #     """
    #     # 如果 shape 是 (1, n)，将其转换为 (n,)
    #     if len(waveform.shape) == 2 and waveform.shape[0] == 1:
    #         waveform = waveform.squeeze(axis=0)  # 去掉第一个维度
    #     # 检查是否为一维数组
    #     if len(waveform.shape) != 1:
    #         raise ValueError(f"Expected waveform to be 1D, but got shape {waveform.shape}")
    #     # 填充或截断到目标长度
    #     if len(waveform) < TARGET_WAVEFORM_LEN:
    #         return np.pad(waveform, (0, TARGET_WAVEFORM_LEN - len(waveform)), mode='constant')
    #     else:
    #         return waveform[:TARGET_WAVEFORM_LEN]
    # def normalize_transcription(transcription):
    #     """
    #     将 transcription 转换为字符串。
    #     """
    #     if isinstance(transcription, np.ndarray):
    #         return transcription.item()  # 转为 Python 字符串
    #     return transcription
    # dataset = dataset.map(operations = normalize_waveform, input_columns = ['waveform'])
    # dataset = dataset.map(operations = normalize_transcription, input_columns = ['transcription'])
    # # 保存为 MindRecord
    # if os.path.exists(OUTPUT_MINDRECORD):
    #     os.remove(OUTPUT_MINDRECORD)
    # dataset.save(OUTPUT_MINDRECORD)



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
    output_file = "/home/wangke/huge_datasets1.6/shuffle/LJS/LJS_mindspore.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time




def mindrecord_memory(pieces):
    import mindspore.dataset as ds
    dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/LJSpeech_short_10w.mindrecord"] * pieces, shuffle=True)

    # count = 0
    # for item in dataset.create_dict_iterator(output_numpy=True):
    #     count += 1
    #     print(f"File name: {item['audio_path']}, Type: {type(item['audio_path'])}")
    # print(f"Mindrecord contains {count} samples!")

    def to_audio(audio_path):
        audio_path = audio_path.tobytes().decode("utf-8").strip()
        audio, sr = librosa.load(audio_path, sr=22050)  # 使用 librosa 加载音频
        return audio

    # dataset = dataset.map(operations=to_audio,input_columns=['audio_path'],output_columns=['audio'])



    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))# 117266  



    batch_size = batch_size_for_all
    paralled_workers = 8
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)

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
    output_file = "/home/wangke/huge_datasets1.6/shuffle/LJS/LJS_indexonly.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time


    



def mindrecord():
    memory_list = []
    throughput_List = []
    time_list = []
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]:#1000w-5e
        memory,throughput,cost_time = mindrecord_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("Mindrecord:")
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
    for i in [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,100000,125000,150000,200000,250000]:#1000w-5e
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
    for i in [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,100000,125000,150000,200000,250000]:#1000w-5e
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
    for i in [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,100000,125000,150000,200000,250000]:#1000w-5e
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
    # 
    # # mindspore_memory(1)
    # mindspore_memory(4)
    # mindspore() 
    # mindrecord()
    # pytorch()
    # tensorflow()
    # mindrecord()

    # mindrecord_memory(1)

    # tensorflow()
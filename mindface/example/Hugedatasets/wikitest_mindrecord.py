
from calendar import c
import os
import time
from PIL import Image
import numpy as np
import gc
import psutil
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


batch_size_for_all = 32

def pytorch_memory(pieces):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from datasets import Dataset
    from torch.utils.data import DataLoader
    import torch
    import torch.multiprocessing as mp

    import torch.distributed as dist
    from torch.utils.data import DataLoader, distributed
    
    with open('/disk1/datasets/wikitext-103/wiki.train.tokens', 'r', encoding='utf-8') as f:
        tokens = [line for line in f]
    tokens = tokens[:100000]
    tokens = tokens * pieces
    dataset = Dataset.from_dict({"text": tokens})
    
    
    batch_size = batch_size_for_all
    paralled_workers = 8

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        #sampler=sampler,
        num_workers=paralled_workers,
        #collate_fn=collate_fn
                )

    print("The number of actual samples: {}".format(100000*pieces)) 


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
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/memory_record/wikitest/wikitest_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time






def tensorflow_memory(pieces):
    import numpy as np
    import tensorflow as tf
    from transformers import AutoTokenizer
    from datasets import load_dataset
    # 读取文本数据
    start_time = tf.timestamp()
    with open('/disk1/datasets/wikitext-103/wiki.train.tokens', 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f]
    # print(len(tokens))
    tokens = tokens[:100000]
    tokens = tokens * pieces

    # print(len(tokens))
    # 创建 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices(tokens)

    # 加载 tokenizer

    # 文本预处理函数
 

    # 应用数据集映射
    
    batch_size = batch_size_for_all
    paralled_workers = 8
    dataset = dataset.batch(batch_size,num_parallel_calls = paralled_workers)
    
    print("The number of actual samples: {}".format(100000*pieces)) 


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
    print("Average throughput of batch at sample number {} is {} items/sec".format(100000*pieces,sum(throughput_list)/len(throughput_list)))

    import csv
    output_file = "/home/wangke/huge_datasets1.6/memory_record/wikitest/wikitest_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time




def mindspore_memory(pieces):
    import mindspore
    import mindspore.dataset as ds
    from mindnlp.transformers import BertTokenizer
    from transformers import AutoTokenizer
    import mindspore.dataset.transforms as transforms
    import mindspore.dataset.text as text
    import mindspore as ms

    # class TextDataset:
    #     def __init__(self, file_path, tokenizer, pieces=1, max_length=128):

    #         self.file_path = file_path
    #         self.pieces = pieces
    #         self.max_length = max_length
            
            
    #         self.tokenizer = tokenizer


    #         # with open(file_path, 'r', encoding='utf-8') as f:
    #         #     self.num_samples = len(f.readlines())  # 计算文件的行数
    #         # print(self.num_samples)
    #         self.lines = 1801350
    #         self.num_samples = pieces * 100000  # 翻倍数据集

    #     def __len__(self):
    #         return self.num_samples

    #     def __getitem__(self, idx):
   
    #         real_idx = idx % self.lines  # 处理数据翻倍
    #         with open(self.file_path, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
    #             token = lines[real_idx].strip()

    #         inputs = self.tokenizer(token, padding='max_length', truncation=True, max_length=self.max_length)

    #         return np.array(inputs['input_ids'])
    #     def data_generator(self):
    #         """生成器，用于生成数据"""
    #         for i in range(self.num_samples):
    #             yield self.__getitem__(i)


    # tokenizer = AutoTokenizer.from_pretrained('/data1/datasets/bert')
    # # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # file_path = '/disk1/datasets/wikitext-103/wiki.train.tokens'
    # dataset = TextDataset(file_path, tokenizer, pieces=pieces)
    # dataset = ds.GeneratorDataset(source=dataset, column_names=["input_ids"])


    with open('/disk1/datasets/wikitext-103/wiki.train.tokens', 'r', encoding='utf-8') as f:
        tokens = [line for line in f]
    tokens = tokens[:100000]
    tokens = tokens * pieces

    dataset = ds.GeneratorDataset(source=tokens, column_names="input")


    # tokens_tensor = np.array(tokens)  # 使用NumPy将tokens转化为数组

    # # 创建TensorDataset
    # dataset = ds.NumpySlicesDataset(tokens_tensor, column_names=["text"])


    # dataset = ds.WikiTextDataset(dataset_dir='/data1/wangcong/wikitext-103/',shuffle=False,num_parallel_workers=8)
    # nums = 0
    # for line in tokens:
    #     nums += 1
    # print(nums)
    
    # temp_file_path = '/data1/wangcong/wiki_train_tokens_temp.txt'
    # with open(temp_file_path, 'w', encoding='utf-8') as f:
    #     f.writelines(tokens)

    # dataset = ds.TextFileDataset(temp_file_path)


    # def data_generator():
    #     for token in tokens:
    #         yield (token,)

    # 使用 MindSpore GeneratorDataset 构建数据集
    # column_names = ["text"]
    # dataset = ds.GeneratorDataset(source=data_generator, column_names=column_names)

    batch_size = batch_size_for_all
    paralled_workers = 8
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)

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
    output_file = "/home/wangke/huge_datasets1.6/memory_record/wikitest/mindspore_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
    return memory,sum(throughput_list)/len(throughput_list), cost_time



def mindrecord_memory(pieces):
    import mindspore.dataset as ds
    dataset = ds.MindDataset(["/disk1/datasets/hugedataset_mr/wikitest.mindrecord"]*pieces, shuffle=True)


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
    output_file = "/home/wangke/huge_datasets1.6/memory_record/wikitest/wikitest_mindrecord.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
    return memory,sum(throughput_list)/len(throughput_list), cost_time




def mindrecord():
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]:#1000w-5e
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
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]:#1000w-5e
        memory,throughput,cost_time = mindspore_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("mindspore:")
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
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]:#1000w-5e#,2000,2500,3000,4000,5000
        memory,throughput,cost_time = tensorflow_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("tensorflow:")
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
    for i in [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]:#1000w-5e
        memory,throughput,cost_time = pytorch_memory(i)
        memory_list.append(memory)
        throughput_List.append(throughput)
        time_list.append(cost_time)
        time.sleep(5); gc.collect()
    print("pytorch:")
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
    # mindspore_memory(args.pieces)
    # pytorch_memory(args.pieces)
    # tensorflow_memory(args.pieces)

    # pytorch_memory(1,8,4)
    # mindspore_memory(1)
    # mindrecord_memory(2)
    # tensorflow_memory(1,8,2)

    # mindspore()
    # pytorch()
    # tensorflow()
    # mindrecord()
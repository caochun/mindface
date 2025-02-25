import psutil
import os
import gc
import time
from mindspore import log as logger

pid = os.getpid()
process = psutil.Process(pid)

batch_size_for_all = 1
pid = os.getpid()
process = psutil.Process(pid)
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="", type=int)
    args = parser.parse_args()
    return args
def memory_log():
    print(process.memory_info().rss / 1024 / 1024,"MB")
    return process.memory_info().rss / 1024 / 1024




def tensorflow_memory(samples):
    dataset = MyDataset(samples)
    return dataset



def tensorflow_memory(samples):
  import tensorflow as tf
  import cv2
  import numpy as np
  import os
  from pycocotools.coco import COCO


  class MyDataset(tf.data.Dataset):
    def __new__(cls, samples):
        # 预先生成一个完整的索引列表
        indices = list(range(samples))
        return tf.data.Dataset.from_tensor_slices(indices)
        

  dataset = MyDataset(samples)

  # tf_dataset = tf_dataset.shuffle(buffer_size=samples)  # 随机打乱
  batch_size = batch_size_for_all
  paralled_workers = 8

  print("The number of actual samples: {}".format(samples))
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
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))


  import csv
  output_file = "/home/wangke/huge_datasets1.6/shuffle/Default/Default_tensorflow.csv"
  mode = 'w' if pieces == 100 else 'a'
  with open(output_file, mode) as file:  # 打开文件以写入
    writer = csv.writer(file)
    writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

  return memory,sum(throughput_list)/len(throughput_list), cost_time




def pytorch_memory(samples):
  import torch
  from torch.utils.data import DataLoader, Dataset

  class MyDataset(Dataset):
    def __init__(self):
      pass
    def __getitem__(self, i):
      return i
    def __len__(self):
      return samples

  dataset = DataLoader(MyDataset(), shuffle=False)

  batch_size = batch_size_for_all
  paralled_workers = 8

  print("The number of actual samples: {}".format(samples))


  # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)
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

    if data_nums == 1000000:
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
  output_file = "/home/wangke/huge_datasets1.6/shuffle/Default/Default_pytorch.csv"
  mode = 'w' if pieces == 100 else 'a'
  with open(output_file, mode) as file:  # 打开文件以写入
    writer = csv.writer(file)
    writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
  return memory,sum(throughput_list)/len(throughput_list), cost_time




def mindspore_memory(samples):
  import mindspore.dataset as ds
  import mindspore as ms
  class MyDataset():
    def __init__(self):
      pass
    def __getitem__(self, i):
      return i
    def __len__(self):
      return samples

  # ms.set_context(mode=ms.GRAPH_MODE)
  # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
  # rank_id = get_rank()
  # rank_size = get_group_size()

  dataset = ds.GeneratorDataset(MyDataset(), "a", shuffle = False)
  samples = dataset.get_dataset_size()
  print("The number of actual samples: {}".format(samples))

  dataset.save("/data1/10w.mindrecord")




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

    if data_nums == 100:
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
  output_file = "/home/wangke/huge_datasets1.6/shuffle/Default/Default_mindspore.csv"
  mode = 'w' if pieces == 100 else 'a'
  with open(output_file, mode) as file:  # 打开文件以写入
    writer = csv.writer(file)
    writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
  return memory,sum(throughput_list)/len(throughput_list), cost_time



def mindrecord_memory(pieces):
  import mindspore.dataset as ds
  import mindspore as ms
  # ms.set_context(mode=ms.GRAPH_MODE)
  # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
  # from mindspore.communication import get_rank, get_group_size
  # from mindspore.communication import init
  # init()
  # rank_id = get_rank()
  # rank_size = get_group_size()
  # print("rank_size = ",rank_size)
  # print("rank_id = ",rank_id)


  # dataset = ds.MindDataset(["/data0/hugedataset_mr/test_mr/1million.mindreocrd"]*pieces, columns_list=["a"], shuffle=False)
  dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/10w.mindrecord"]*pieces, columns_list=["a"], shuffle=True)
  samples = dataset.get_dataset_size()
  print("The number of actual samples: {}".format(samples))


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
  output_file = "/home/wangke/huge_datasets1.6/shuffle/Default/Default_mindrecord.csv"
  mode = 'w' if pieces == 11 else 'a'
  with open(output_file, mode) as file:  # 打开文件以写入
    writer = csv.writer(file)
    writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])
    
  return memory,sum(throughput_list)/len(throughput_list), cost_time




def tensorflow():#1000000
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [10000000,20000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000,110000000,120000000,130000000,140000000,150000000,200000000,250000000,300000000,400000000,500000000]:#1000w-5e
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

def pytorch():#1000000
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [10000000,20000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000,110000000,120000000,130000000,140000000,150000000,200000000,250000000,300000000,400000000,500000000]:#1000w-5e
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

def mindspore():#1000000
    memory_list = []
    throughput_List = []
    time_list = []
    for i in [10000000,20000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000,110000000,120000000,130000000,140000000,150000000,200000000,250000000,300000000,400000000,500000000]:#1000w-5e
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

def mindrecord():#1000000
    memory_list = []
    throughput_List = []
    time_list = []

    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]
    # [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,250,300,400,500]
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]:#1000w-5e
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

if __name__ == "__main__":
  args = parse_args()
  # tensorflow_memory(args.pieces)
  mindrecord_memory(args.pieces)
  # pytorch_memory(110000000)
  # mindspore_memory(100000)
  # mindrecord_memory(110)
  # mindrecord()
  # mindspore()
  # pytorch()
  # tensorflow()
# def write_mr():
#   import mindspore.dataset as ds
#   from mindspore.mindrecord import FileWriter

#   class MyDataset():
#     def __init__(self):
#       pass
#     def __getitem__(self, i):
#       return i
#     def __len__(self):
#       return 1000000

#   dataset = ds.GeneratorDataset(MyDataset(), "a", shuffle=True)
#   logger.warning(f"{dataset.output_shapes()}, {dataset.output_types()}")
#   import pdb; pdb.set_trace()
#   #dataset.save("/data0/test_mr/1million.mindreocrd")

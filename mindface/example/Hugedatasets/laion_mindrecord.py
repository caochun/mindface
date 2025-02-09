import mindspore.dataset as ds
import time
import numpy as np
import requests
import os
import gc
from io import BytesIO
from PIL import Image
import pyarrow.parquet as pq
file_path = '/disk1/datasets/laion2b'
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



#5000#8319
import matplotlib.pyplot as plt


batch_size_for_all = 32



def pytorch_memory(pieces):
    import os
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    from PIL import Image
    import numpy as np
    import time
    import torchvision
    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/disk1/datasets/bert')

    batch_size = batch_size_for_all
    dataset_num_workers = 8

    # 2. 自定义数据集
    import os
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

# 初始化 tokenizer 和 transform
    tokenizer = AutoTokenizer.from_pretrained("/disk1/datasets/bert")  # 示例 tokenizer
    transform = lambda x: np.array(x)  # 示例 transform

    class ImageTextDataset(Dataset):
        def __init__(self, image_dirs, text_dirs, pieces):
            self.image_paths = []
            self.texts = []
            self.pieces = pieces
            # 读取图像和文本数据
            for idx in range(16):  # 假设有8个文件夹，每个文件夹包含1250个数据
                image_folder = os.path.join(image_dirs, f"image{idx}")
                text_file = os.path.join(text_dirs, f"text{idx}.txt")
                
                # 读取对应的文本文件
                with open(text_file, 'r') as txt_f:
                    texts = txt_f.readlines()
                
                # 读取图片文件夹中的所有图片路径
                image_files = sorted(os.listdir(image_folder)) # 读取所有图片
                
                # 遍历图片文件夹中的图片并将路径和文本关联起来
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(image_folder, image_file)
                    text = texts[i].strip()  # 获取对应的文本描述
                    self.image_paths.append(image_path)
                    self.texts.append(text)
                self.image_paths_now = self.image_paths[:5000] * pieces
                self.texts_now = self.texts[:5000] * pieces

        def __len__(self):
            """返回数据集的大小"""
            return len(self.image_paths_now)
        
        def __getitem__(self, idx):
            """
            根据索引返回图片和对应的描述
            
            :param idx: 索引
            :return: 图片和描述的元组 (image, text)
            """
            img_path = self.image_paths_now[idx]
            text = self.texts_now[idx]
            
            # 处理文本：Tokenize 并 Padding
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
            
            # 打开并处理图片
            img = Image.open(img_path).convert("RGB")  # 打开图片并转换为RGB模式
            #img = np.array(img)
            
            # 转换为 numpy 数组并转为 PyTorch 张量
            
            img = transform(img)
            img = np.array(img)
            # 返回图片和文本的张量
            return img, inputs


    # 3. 创建 DataLoader
    image_dirs = '/disk1/datasets/laion2b/image_clean/'  # 假设图片目录
    text_dirs = '/disk1/datasets/laion2b'  # 假设文本目录
    
    dataset = ImageTextDataset(image_dirs, text_dirs, pieces)
        
    data_loader = DataLoader(dataset, shuffle=False, num_workers= 1 )
        


    # 4. 示例使用


    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        
    ])



    data_loader = create_data_loader(image_dirs, text_dirs, batch_size=batch_size, num_workers=dataset_num_workers , pieces=pieces)
    
    # 5. 训练循环
    print("The number of actual samples: {}".format(5000*pieces)) 

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
    print("Average throughput of batch at sample number {} is {} items/sec".format(5000*pieces,sum(throughput_list)/len(throughput_list)))
        # print("Average throughput of batch {i} at sample number {} is {} items/sec".format(i, samples, throughput_list[i]))
    import csv
    output_file = "/home/wangke/huge_datasets1.6/memory_record/laion/laion_pytorch.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time




def tensorflow_memory(pieces):
    import os
    import tensorflow as tf
    from transformers import AutoTokenizer
    from PIL import Image
    import numpy as np
    import time

    batch_size = batch_size_for_all
    dataset_num_workers = 8

    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/disk1/datasets/bert')

    # 2. 自定义数据集
    class ImageTextDataset():
        def __init__(self, image_dirs, text_dirs, pieces):
            self.image_paths = []
            self.texts = []
            self.pieces = pieces
            # 读取图像和文本数据
            for idx in range(16):  # 假设有8个文件夹，每个文件夹包含1250个数据
                image_folder = os.path.join(image_dirs, f"image{idx}")
                text_file = os.path.join(text_dirs, f"text{idx}.txt")
                
                # 读取对应的文本文件
                with open(text_file, 'r') as txt_f:
                    texts = txt_f.readlines()
                
                # 读取图片文件夹中的所有图片路径
                image_files = sorted(os.listdir(image_folder))  # 读取所有图片
                
                # 遍历图片文件夹中的图片并将路径和文本关联起来
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(image_folder, image_file)
                    text = texts[i].strip()  # 获取对应的文本描述
                    self.image_paths.append(image_path)
                    self.texts.append(text)
                self.image_paths_now = self.image_paths[:5000] * pieces
                self.texts_now = self.texts[0:5000] * pieces
        def __len__(self):
            """返回数据集的大小"""
            return len(self.image_paths_now)

        def __getitem__(self, idx):
            """
            根据索引返回图片和对应的描述
            """
            img_path = self.image_paths_now[idx]
            text = self.texts_now[idx]
            
            # 处理文本：Tokenize 并 Padding
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128)
            
            # 打开并处理图片
            img = Image.open(img_path).convert("RGB")  # 打开图片并转换为RGB模式
            img = np.array(img)
            
            return img, inputs['input_ids']

    # 3. 数据预处理函数
    def image_processing(img):
        img = tf.image.resize(img, (128, 128))
        return img


    def create_data_loader(image_dirs, text_dirs, batch_size=32, num_workers=4,pieces = pieces):
        dataset = ImageTextDataset(image_dirs, text_dirs, pieces)


        tf_dataset = tf.data.Dataset.from_generator(generator,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),  # 图像的 shape
                                                        tf.TensorSpec(shape=(128,), dtype=tf.int32)  # 文本的 tokenized 输出
                                                    ))

        # 设置批次和并行处理
        tf_dataset = tf_dataset.batch(batch_size,num_parallel_calls=num_workers)

        return tf_dataset

    # 5. 示例使用
    image_dirs = '/disk1/datasets/laion2b/image_clean/'  # 假设图片目录
    text_dirs = '/disk1/datasets/laion2b'  # 假设文本目录


    dataset = create_data_loader(image_dirs, text_dirs, batch_size=batch_size, num_workers=dataset_num_workers,pieces = pieces)

    dataset = dataset.batch(batch_size)
    print("The number of actual samples: {}".format(5000*pieces))# 117266  
    # dataset = dataset.batch(batch_size,num_parallel_workers=num_parallel_workers)



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
    output_file = "/home/wangke/huge_datasets1.6/memory_record/laion/laion_tensorflow.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time



    





def mindspore_memory(pieces):
    import pyarrow.parquet as pq
    from mindspore import dataset as ds
    from transformers import AutoTokenizer
    import mindspore.dataset as ds
    



    tokenizer = AutoTokenizer.from_pretrained('/disk1/datasets/bert')
    # 读取 Parquet 文件的元数据


    class ImageTextDataset:
        def __init__(self, image_dirs, text_dirs,pieces):
       
            self.image_paths = []
            self.texts = []
            self.pieces = pieces
            for idx in range(16):  # 假设有8个文件夹，每个文件夹包含1250个数据
                image_folder = os.path.join(image_dirs, f"image{idx}")
                text_file = os.path.join(text_dirs, f"text{idx}.txt")
                
                # 读取对应的文本文件
                with open(text_file, 'r') as txt_f:
                    texts = txt_f.readlines()
                
                # 读取图片文件夹中的所有图片路径
                image_files = sorted(os.listdir(image_folder)) # 读取前 num_images 个图片
                
                # 遍历图片文件夹中的图片并将路径和文本关联起来
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(image_folder, image_file)
                    text = texts[i].strip()  # 获取对应的文本描述
                    self.image_paths.append(image_path)
                    self.texts.append(text)

                self.image_paths_now = self.image_paths[:5000] * pieces
                self.texts_now = self.texts[0:5000] * pieces
        def __len__(self):
            """返回数据集的大小"""
            return len(self.image_paths_now)
        
        def __getitem__(self, idx):
            """
            根据索引返回图片和对应的描述
            
            :param idx: 索引
            :return: 图片和描述的元组 (image, text)
            """
            img_path = self.image_paths_now[idx]
            text = self.texts_now[idx]
            inputs = tokenizer(text, 
                                padding='max_length', truncation=True, max_length=256)
            # 打开图片
            #img = Image.open(img_path)
            #img = np.fromfile(img_path, np.uint8)
            return img_path, inputs['input_ids']

    def read_img(img_path):
        """
        import pdb
        pdb.set_trace() 
        """
         # 在这里设置断点
        img = np.fromfile(img_path.item(), np.uint8)
        img=ds.vision.Decode()(img)
        # plt.imshow(img)
        return img


    batch_size = batch_size_for_all
    paralled_workers = 8

    # 创建 GeneratorDataset
    def generate_image_text_dataset(image_dirs, text_files, pieces):
        dataset = ImageTextDataset(image_dirs, text_files, pieces)
        
        # 传递给 GeneratorDataset
        return ds.GeneratorDataset(source=dataset, column_names=["image", "text"], 
                                   num_parallel_workers=paralled_workers)

    # 示例使用
    image_path='/disk1/datasets/laion2b/image_clean/'
    file_path='/disk1/datasets/laion2b' 
    dataset = generate_image_text_dataset(image_path,file_path,pieces)
    dataset=dataset.map(operations=read_img,#
                        input_columns="image",num_parallel_workers=paralled_workers)

    dataset=dataset.map(operations=[ds.vision.Resize([128,128])],
                        input_columns="image",num_parallel_workers=paralled_workers)
                        

    # for idx,data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
    #     print(data['image'].shape)
    #     print(data['text'].shape)

    # dataset.save('/data0/hugedataset_mr/laion5000.mindrecord')
 


    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))# 117266  

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
    output_file = "/home/wangke/huge_datasets1.6/memory_record/laion/laion_mindspore.csv"
    mode = 'w' if pieces == 100 else 'a'
    with open(output_file, mode) as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow([memory,sum(throughput_list)/len(throughput_list),cost_time])

    return memory,sum(throughput_list)/len(throughput_list), cost_time




    
def mindrecord_memory(pieces):
    import mindspore.dataset as ds
    dataset = ds.MindDataset(["/home/wangke/wangke_indexonly/laion.mindrecord"]*pieces, shuffle=True)






    samples = dataset.get_dataset_size()
    print("The number of actual samples: {}".format(samples))#

    batch_size = batch_size_for_all
    paralled_workers = 8
    # dataset = dataset.batch(batch_size,num_parallel_workers=paralled_workers)



    data_nums = 0
    start = time.time()
    throughput_list = []
    throughput_start = time.time()
    for _ in dataset.create_dict_iterator():
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
    output_file = "/home/wangke/huge_datasets1.6/memory_record/laion/laion_indexonly.csv"
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
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#1000w-5e
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
    #[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,4000,5000]
    for i in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,40000,50000,60000,80000,100000]:#1000w-5e
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
    # mindspore_memory(args.pieces)
    # pytorch_memory(args.pieces)
    # tensorflow_memory(args.pieces)

    # mindrecord_memory(1)

    # mindrecord()
    # mindspore()
    # tensorflow()
    # pytorch()
    # mindspore()
    # mindspore_memory(1)
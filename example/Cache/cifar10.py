import os
import time
import os

import torch

# import tensorflow as tf
import mindspore.dataset as ds

data_dir = "/home/wangcong/cifar-10-tf/"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_data_nums = 500
import csv
batch_sizes = [32, 64, 128, 256, 512]
def write_list_to_csv(data_list, file_path):
    """
    将一个列表写入 CSV 文件并保存到指定路径。

    参数:
    - data_list: 要写入 CSV 文件的列表，列表中的每个元素可以是子列表（代表行数据）。
    - file_path: 保存 CSV 文件的完整路径。
    """
    # 打开指定路径的文件，使用 'w' 模式表示写入，如果文件不存在则创建
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 直接写入一行数据
        writer.writerow(data_list)
def mindspore_offload(batch_size, dataset_num_workers=8, cache1=0,cache2=0):
    data_nums = 0
    import mindspore as ms
    from mindspore.communication import get_rank, get_group_size
    from mindspore.communication import init
    
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()
    rank_id = get_rank()
    rank_size = get_group_size()
    #start_time = time.time()
    
    # dataset = ds.ImageFolderDataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_workers, decode=False)
    data_dir = "/nfs/dataset/workspace/mindspore_dataset/cifar-10-batches-bin/"
    
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=1020348693, size=0, spilling=False)
        dataset = ds.Cifar10Dataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_workers, shuffle=False,cache=test_cache_1
                                    ,num_shards=rank_size, shard_id=rank_id)
    else :
        dataset = ds.Cifar10Dataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_workers, shuffle=False
                                    ,num_shards=rank_size, shard_id=rank_id)
    
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=1020348693, size=0, spilling=False)
    
        dataset = dataset.map(operations=[ds.vision.Resize([256,256]),
                                          ds.vision.Normalize(mean=mean, std=std),
                                          ds.vision.HWC2CHW()], input_columns="image", 
                                          num_parallel_workers=dataset_num_workers,
                                          cache=test_cache_2)
        

    else:
        dataset = dataset.map(operations=[ds.vision.Resize([256,256]),
                                          ds.vision.Normalize(mean=mean, std=std),
                                          ds.vision.HWC2CHW()], input_columns="image", 
                                          num_parallel_workers=dataset_num_workers,
                                          )
    dataset = dataset.batch(batch_size)
    time_all=[]
    for i in range(8):
        
            
        start_time = time.time()
       
        for _ in dataset.create_tuple_iterator(output_numpy=True):
            data_nums = data_nums + 1
           
        end_time = time.time()
        cost_time = end_time - start_time
        time_all.append(cost_time)
        print("mindspore cost time is: " + str(cost_time))
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}cifar10.csv")
    return cost_time, dataset

def pytorch_imagefolder(batch_size, dataset_num_workers=8):
    data_nums = 0
    import torchvision

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
       
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    # 下载并加载训练集
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                               num_workers=dataset_num_workers)
    for epoch in range(2):
        start_time = time.time()
        for repeat in range(1):
            for _ in train_loader:
                data_nums = data_nums + 1
                # if data_nums > test_data_nums:
                #     break
            
        end_time = time.time()
        cost_time = end_time - start_time
        print("pytorch cost time is: " + str(cost_time))
    return cost_time

def tensorflow_imagefolder(batch_size, dataset_num_workers=8):
    import numpy as np
    import pickle
    import tensorflow as tf
    start_time = time.time()
    def load_cifar10_batch(file):
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            images = dict[b'data']
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 调整形状为 [batch_size, 32, 32, 3]
            labels = np.array(dict[b'labels'])
        return images, labels

    def load_cifar10_data(data_dir):
        """加载所有CIFAR-10数据"""
        # 训练数据
        images_train = []
        labels_train = []
        for i in range(1, 6):
            file = os.path.join(data_dir, f'data_batch_{i}')
            images, labels = load_cifar10_batch(file)
            images_train.append(images)
            labels_train.append(labels)
        
        images_train = np.concatenate(images_train)
        labels_train = np.concatenate(labels_train)        
        return (images_train, labels_train)

    def random_sharpness(image, factor=0.5):
        image = tf.cast(image, tf.float32)
        blurred_image = tf.image.adjust_contrast(image, contrast_factor=0.5)
        sharp_image = tf.add(tf.multiply(factor, image), tf.multiply(1.0-factor, blurred_image))

        sharp_image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
                              lambda: sharp_image,
                              lambda: image)
        return sharp_image
    
    def augment(image):
        #image = tf.image.random_flip_up_down(image)
        #image = random_sharpness(image, factor=1.0)
        image = tf.image.resize(image, [256, 256])
        image = (tf.cast(image, tf.float32) - mean) / std
        image = tf.transpose(image, [2, 0, 1])
        return image
    
    # 加载数据
    
    (train_images, train_labels) = load_cifar10_data("/home/wangcong/cifar-10-tf/cifar-10-batches-py/")

    # 将数据转换为 TensorFlow 数据集
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.map( augment,num_parallel_calls=dataset_num_workers)
    train_dataset = train_dataset.batch(batch_size)

    data_nums = 0
    for epoch in range(2):
        start_time = time.time()
        for repeat in range(1):
            for _ in train_dataset:
                data_nums = data_nums + 1
                # if data_nums > test_data_nums:
                #     break
            
        end_time = time.time()
        cost_time = end_time - start_time
        print("tesnsorflow cost time is: " + str(cost_time))
    return cost_time
 
if __name__ == "__main__":
    # 测试各个
    # test_ms_autotune()

    batch_sizes = [32]
    workers = [8]

    bs = 32
    num_worker = 8
    # for num_worker in workers:
    for num_worker in workers:

    
        print("current worker: " + str(num_worker))
        
        #pytorch_imagefolder(bs, num_worker)
        #tensorflow_imagefolder(bs, num_worker)
        mindspore_offload(bs, num_worker, 0,0)
        #mindspore_offload(bs, num_worker, 1,0)
        #mindspore_offload(bs, num_worker, 0,1)
        #mindspore_offload(bs, num_worker, 1,1)


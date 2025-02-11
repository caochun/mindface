import os
import time

import torch

# import tensorflow as tf
import os

import mindspore.dataset as ds

data_dir =  "/nfs/dataset/workspace/mindspore_dataset/imagenet/imagenet_original/train"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# test_data_nums = 1000
sample_nums = 50000
from mindspore.dataset import Cifar10Dataset, Cifar100Dataset, DistributedSampler
batch_sizes = [32, 64, 128, 256, 512]

def mindspore_offload(batch_size, dataset_num_workers=8, cache1=0,cache2=0):
    import mindspore as ms
    
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    import csv
    """
    def write_list_to_csv(data_list, file_path):
        
        # 打开指定路径的文件，使用 'w' 模式表示写入，如果文件不存在则创建
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # 直接写入一行数据
            writer.writerow(data_list)
    
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()
    rank_id = get_rank()
    rank_size = get_group_size()
    """
    data_nums = 0
    total_time=0
    first_time_start=time.time()
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=99366030, size=0)
        dataset = ds.ImageFolderDataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_workers, 
                                        cache=test_cache_1,shuffle=False,decode=True
                                        )
    else :
        dataset = ds.ImageFolderDataset(dataset_dir=data_dir,num_parallel_workers=dataset_num_workers,shuffle=False,decode=True
                                        )
       
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=3073373396, size=0, spilling=True)
    
        dataset = dataset.map(operations=[ ds.vision.Resize((256, 256)),ds.vision.Normalize(mean=mean, std=std)], input_columns="image", 
                          num_parallel_workers=dataset_num_workers,cache=test_cache_2)
        

    else:
       dataset = dataset.map(operations=[ ds.vision.Resize((256, 256)),ds.vision.Normalize(mean=mean, std=std)], input_columns="image", 
                          num_parallel_workers=dataset_num_workers)
    
    
    
    
    dataset = dataset.batch(batch_size,num_parallel_workers=dataset_num_workers)
    num_epochs=8
    interator=dataset.create_tuple_iterator(output_numpy=True)
    time_all=[]
    for i in range(num_epochs):
        data_nums = 0
        if i == 0:
            first_time_start=time.time()
        else:
            epoch_start_time = time.time() 
        for _ in interator:
            data_nums += 1
            
            
        if i == 0:
            first_time_end = time.time()
            first_epoch_time = first_time_end - first_time_start
            time_all.append(first_epoch_time)
            print(f"First epoch time: {first_epoch_time} seconds")
        else:
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            time_all.append(epoch_time)
            print(f'epoch{i}',epoch_time)
            total_time += epoch_time
    #write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}imagenet.csv")
# Calculate average time for subsequent epochs
    if num_epochs>1:
        average_epoch_time = total_time / (num_epochs - 1)
        print(f"Average time for subsequent epochs: {average_epoch_time} seconds")
    return  dataset


   

def pytorch_imagefolder(batch_size, dataset_num_workers=8):
    data_nums = 0
    import torchvision
    total_time=0
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        #torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,hue=0.1),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.RandomAdjustSharpness(1),
        #torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.ImageFolder(root=data_dir, transform=transform),
        num_workers=dataset_num_workers,
        multiprocessing_context=None,
        batch_size=batch_size
    )
    num_epochs=3
    
    for i in range(2) :
        data_nums = 0
        start = time.time()
        for  _ in train_loader:
            #print(images.shape)  # 输出图像的tensor形状
            #print(targets)        # 输出目标的标注信息
            data_nums+=1
            
        end = time.time()
        print("torch reading time", end-start)
        print('\n')

def tf_imagefolder(batch_size, dataset_num_workers=8):
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    # image_size = 224
    data_nums = 0
   
    
    train_filenames = [os.path.join(data_dir, classname, filename)
                       for classname in os.listdir(data_dir)
                       for filename in os.listdir(os.path.join(data_dir, classname))]
    
    def random_sharpness(image, factor=0.5):
        image = tf.cast(image, tf.float32)
        blurred_image = tf.image.adjust_contrast(image, contrast_factor=0.5)
        sharp_image = tf.add(tf.multiply(factor, image), tf.multiply(1.0-factor, blurred_image))

        sharp_image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
                              lambda: sharp_image,
                              lambda: image)
        return sharp_image
    
    def random_color_adjust(image):
        image = tf.image.random_brightness(image, max_delta=0.2)  # 随机调整亮度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 随机调整对比度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 随机调整饱和度
        image = tf.image.random_hue(image, max_delta=0.1)  # 随机调整色相
        return image
    
    def parse_function(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [256, 256])
        #image = tf.image.resize(image, [224, 224])
        #image = random_color_adjust(image)
        #image = tf.image.random_flip_up_down(image)
        #image = random_sharpness(image, factor=1.0)
        image = (tf.cast(image, tf.float32) - mean) / std
        image = tf.transpose(image, [2, 0, 1])
        return image

    train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
    train_dataset = train_dataset.map(parse_function, num_parallel_calls=dataset_num_workers)
    train_dataset = train_dataset.batch(batch_size)
    
    for epoch in range(2):
        start_time = time.time()
        data_nums = 0
        for _ in train_dataset:
            data_nums = data_nums + 1
            
        end_time = time.time()
        cost_time = end_time - start_time
        print("tensorflow cost time is: " + str(cost_time))
        

if __name__ == "__main__":
    # 测试各个
    # test_ms_autotune()
   
    # num_worker = 32
    workers = [12]
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    bs = 32
    
    # for num_worker in workers:
    for num_worker in workers:
        print("current worker: " + str(num_worker))
        # print("current worker: " + str(num_worker))
        #tf_imagefolder(bs, num_worker)
        #pytorch_imagefolder(bs, num_worker)
        #mindspore_offload(bs, num_worker, 0,0)
        #mindspore_offload(bs, num_worker, 1,0)
        mindspore_offload(bs, num_worker, 0,1)


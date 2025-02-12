import time
from pycocotools.coco import COCO
import os

from PIL import Image
import os
import gc
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size
data_dir = '/home/wangcong/coco-zr/coco2017'
ann_file_train = f'{data_dir}/annotations/instances_train2017.json'
img_dir_train = f'{data_dir}/train2017/'

from PIL import Image
from pycocotools.coco import COCO
from mindspore import log as logger
import numpy as np
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sample_nums = 100
import csv
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


def pytorch_imagefolder(bs, num_worker):
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.datasets import CocoDetection
    import time
    import torch.distributed as dist
    from torch.utils.data import DataLoader, distributed
    dist.init_process_group(backend='gloo', init_method='env://')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    def collate_fn(batch):
        # 提取图像和目标
        images, targets = zip(*batch)
        max_num_objects = max(len(target) for target in targets)  # 找到最大目标数量
        padded_boxes = []
        padded_labels = []
        
        for target in targets:
            bboxes = []
            categories = []
            for obj in target:
                bboxes.append(obj['bbox'])  # 提取bbox
                categories.append(obj['category_id'])  # 提取类别
            if len(bboxes)==0:
                bboxes = np.zeros((0, 4), dtype=np.float32)
            else: 
                bboxes=np.array(bboxes)
            # 直接在这里进行padding
            padded_box = (np.pad(bboxes, ((0, max_num_objects - len(bboxes)), (0, 0)), mode='constant', constant_values=-1))
            padded_label = (np.pad(categories, (0, max_num_objects - len(categories)), mode='constant', constant_values=-1))
            padded_boxes.append(padded_box)  
            padded_labels.append(padded_label)     
        return images, np.array(padded_boxes),np.array(padded_labels)
    coco_train = CocoDetection(root=img_dir_train,
                               annFile=ann_file_train,
                               transform=transform)
    sampler = distributed.DistributedSampler(coco_train)
    
    # 创建DataLoader
    train_loader = DataLoader(dataset=coco_train,
                            batch_size=bs,
                            shuffle=True,
                            num_workers=num_worker,
                            collate_fn=collate_fn,
                            sampler=sampler)
    
    # 测试DataLoader
    now_sample=0
    for i in range(4) :
        start = time.time()
        for  _ in train_loader:
            #print(images.shape)  # 输出图像的tensor形状
            #print(targets)        # 输出目标的标注信息
            now_sample+=1
            
        end = time.time()
        print("torch reading time", end-start)
        print('\n')

def mindspore(batch_size, dataset_num_workers=8, cache1=0,cache2=0):
    import mindspore.dataset as ds
    import mindspore as ms
    import mindspore.dataset.vision as vision
    import mindspore.dataset.transforms as transforms
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()
    rank_id = get_rank()
    rank_size = get_group_size()
    total_time=0
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=2733804946, size=0, spilling=False)
        dataset = ds.CocoDataset(dataset_dir=img_dir_train,
                         annotation_file=ann_file_train,
                         task='Detection',
                         decode=True,
                         num_parallel_workers=dataset_num_workers,
                         cache=test_cache_1,
                         num_shards=rank_size, shard_id=rank_id)
    else :
        dataset = ds.CocoDataset(dataset_dir=img_dir_train,
                         annotation_file=ann_file_train,
                         task='Detection',
                         decode=True,
                         num_parallel_workers=dataset_num_workers,
                         num_shards=rank_size, shard_id=rank_id)
       
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=2933706385, size=0, spilling=False)
        dataset = dataset.map(operations=[vision.Resize((256,256)),vision.Normalize(mean=mean, std=std)],
                          input_columns="image", num_parallel_workers=dataset_num_workers,cache=test_cache_2)
        

    else:
       dataset = dataset.map(operations=[vision.Resize((256,256)),vision.Normalize(mean=mean, std=std)],
                          input_columns="image", num_parallel_workers=dataset_num_workers)
    dataset=dataset.project(['image', 'bbox', 'category_id'])
    dataset = dataset.padded_batch(batch_size, drop_remainder=True,num_parallel_workers=dataset_num_workers,pad_info={"bbox": (None, -1),"category_id": (None, -1)})

    num_epochs=10
    interator=dataset.create_tuple_iterator(output_numpy=True)
    #next(interator)
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
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}coco.csv")
# Calculate average time for subsequent epochs
    if num_epochs>1:
        average_epoch_time = total_time / (num_epochs - 1)
        print(f"shard_id{rank_id}Average time for subsequent epochs: {average_epoch_time} seconds")
    return  dataset

def tf_imagefolder(bs, num_worker):
    import time
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    from pycocotools.coco import COCO
    
    # 设置路径
    # 加载COCO注释
    coco = COCO(ann_file_train)
    mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    # 获取所有图像的ID
    image_ids = coco.getImgIds()

    # 自定义数据生成器
    def data_generator(image_ids, coco, img_dir_train):
        for image_id in image_ids:
            # 加载图像数据
            image_data = coco.loadImgs(image_id)[0]
            image_path = os.path.join(img_dir_train, image_data['file_name'])
            #image = cv2.imread(image_path)
            
            # 获取图像的注释
            annotation_ids = coco.getAnnIds(imgIds=image_data['id'])
            annotations = coco.loadAnns(annotation_ids)
            
            # 提取边界框和标签
            bboxes = []
            labels = []
            for annotation in annotations:
                bbox = annotation['bbox']
                label = annotation['category_id']
                bboxes.append(bbox)
                labels.append(label)
            
            # 转换为numpy数组
            if bboxes:
                bboxes = np.array(bboxes, dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)  # 保持形状一致
            labels = np.array(labels, dtype=np.int32)
            
            yield image_path, bboxes, labels

    # 创建TensorFlow数据集
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_ids, coco, img_dir_train),
        output_signature=(
            tf.TensorSpec(shape=(None), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    def preprocess_image(image, bboxes, labels, target_size=(256, 256)):
        # 调整图像大小
        
        image_string = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, target_size)
        
        # 归一化图像
        image = tf.cast(image, tf.float32) / 255.0
       
        image = (image - mean) / std
        image = tf.transpose(image, [2, 1, 0])
        return image, bboxes, labels

    # 应用预处理并分批
    dataset = dataset.map(preprocess_image, num_parallel_calls=num_worker)
    dataset = dataset.padded_batch(
    batch_size=bs,
    padded_shapes=(
        (3,256, 256),  # 图像的形状
        (None, 4),  # bbox的形状
        (None,)  # labels的形状
    ),
    padding_values=(0.0, -1.0, -1)  # 填充值
)

    for epoch in range(2):
        start = time.time()
        for i, (images, bboxes, labels) in enumerate(dataset):
            #print(images.shape)  # 输出图像的tensor形状
            #print(targets)        # 输出目标的标注信息
            _, _ = images, bboxes
            
        end = time.time()
        print("tensorflow reading time", end-start)
        print('\n')

if __name__ == "__main__":
    # 测试各个
    # test_ms_autotune()
    batch_sizes = [32]
    # num_worker = 32
    workers = [8]

    bs = 32
    num_worker = 8
    # for num_worker in workers:
    for num_worker in workers:
        print("current worker: " + str(num_worker))
        # print("current worker: " + str(num_worker))
        pytorch_imagefolder(bs,num_worker)
        
        #tf_imagefolder(bs, num_worker)
        
        #mindspore(bs, num_worker, 0,0)
        #mindspore(bs, num_worker, 1,0)
        #mindspore(bs, num_worker, 0,1)

import os
import time

import torch


from torch.utils.data import DataLoader, Dataset


import xml.etree.ElementTree as ET
import os
import time
from PIL import Image
import numpy as np
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
label_map = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'cat': 6,
    'chair': 7,
    'cow': 8,
    'diningtable': 9,
    'dog': 10,
    'horse': 11,
    'motorbike': 12,
    'person': 13,
    'pottedplant': 14,
    'sheep': 15,
    'sofa': 16,
    'train': 17,
    'tvmonitor': 18,
    'car':19,
    'bus':20
}
def pytorch_imagefolder(batch_size, dataset_num_workers=8):
    import torchvision
    def collate_fn(batch):
        imgs, targets = zip(*batch)

        boxes = []
        labels = []

        for target in targets:
            
            b = []
            l = []
            for obj in target['annotation']['object']:
                bbox = obj['bndbox']
                b.append([
                    int(float(bbox['xmin'])),
                    int(float(bbox['ymin'])),
                    int(float(bbox['xmax'])),
                    int(float(bbox['ymax']))
                ])
                label_name = obj['name']
                l.append(label_map[label_name])

            boxes.append(np.array(b))
            labels.append(np.array(l))

        # 找到最大的 box 数量
        max_boxes = max(b.shape[0] for b in boxes)

        # 填充 boxes 和 labels
        padded_boxes = np.array([np.pad(b, ((0, max_boxes - b.shape[0]), (0, 0)), mode='constant', constant_values=-1) for b in boxes])
        padded_labels = np.array([np.pad(l, (0, max_boxes - l.shape[0]), mode='constant', constant_values=-1) for l in labels])

        return torch.stack(imgs), torch.tensor(padded_boxes), torch.tensor(padded_labels)
        
    
    data_nums = 0
    

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.RandomAdjustSharpness(1),
        #torchvision.transforms.RandomVerticalFlip(),
        #torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    # 下载并加载训练集
    train_dataset = torchvision.datasets.VOCDetection(
        root='/home/wangcong/datasets/VOC',
        year='2012',
        image_set='train',
        download=False,
        transform=transform,
        
        
    ) 
  
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                               num_workers=dataset_num_workers,collate_fn=collate_fn)
    for epoch in range(2):
        start_time = time.time()
        for _ in train_loader:
            data_nums = data_nums + 1
            # if data_nums > test_data_nums:
            #     break
        
        end_time = time.time()
        cost_time = end_time - start_time
        print("pytorch cost time is: " + str(cost_time))
    return cost_time
def tensorflow_imagefolder(batch_size, dataset_num_workers=8):
    
    import tensorflow as tf
    
            
       
    def parse_voc_xml(filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        
        boxes = []
        labels = []
        all_box=[]
        
        for obj in root.findall('object'):
            boxes = []
            bbox = obj.find('bndbox')
            for location in list(bbox):
                boxes.append(
                int(float(location.text))
                )
                #import pdb;
                #pdb.set_trace() 
            label_name =list(obj)[0].text
            labels.append(label_map[label_name])
            all_box.append(boxes)
            #print( all_box)
        
        return all_box, labels

    def load_voc_data(img_dir, ann_dir):
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            ann_path = os.path.join(ann_dir, img_file.replace('.jpg', '.xml'))
            # 读取图像
            #print(img_path)
            #image_string = tf.io.read_file(img_path)
            #image = tf.image.decode_jpeg(image_string, channels=3)
            #img = cv2.resize(img, (256, 256))  
            #import pdb;pdb.set_trace()
            
            # 读取标注
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, labels = parse_voc_xml(ann_path)

            yield img_path, boxes, labels
    

    # 设置参数
    batch_size = 32  # 自定义批次大小
    img_dir = '/home/wangcong/datasets/VOC/VOCdevkit/VOC2012/JPEGImages'
    ann_dir = '/home/wangcong/datasets/VOC/VOCdevkit/VOC2012/Annotations'
   
    def preprocess_image(image, bboxes, labels, target_size=(256, 256)):
        image_string = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image_string, channels=3)
        # 调整图像大小
        image = tf.image.resize(image, target_size)
        #import pdb;pdb.set_trace()
        return image, bboxes, labels

    # 使用从生成器创建数据集
    dataset = tf.data.Dataset.from_generator(
        lambda: load_voc_data(img_dir, ann_dir),
        output_signature=(
            tf.TensorSpec(shape=(None), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.int64),  # boxes
            tf.TensorSpec(shape=(None,), dtype=tf.int64)     # labels
        )
    )
    dataset = dataset.map(preprocess_image, num_parallel_calls=dataset_num_workers) 
    dataset = dataset.padded_batch(batch_size, padded_shapes=([256, 256, 3], [None, 4], [None]))

    # 测试数据加载
    for epoch in range(2):
        start_time = time.time()
        data_nums = 0
        for _ in dataset:
            data_nums += 1

        end_time = time.time()
        cost_time = end_time - start_time
        print("TensorFlow cost time is: " + str(cost_time))

def mindspore(batch_size, paralled_workers=8, cache1=0,cache2=0):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore as ms
    
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    import csv
    
    data_nums=0
    
    
    
    rescale_op = vision.Resize([256,256])
    
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=1213414930, size=0, spilling=True)
        dataset = mindspore.dataset.VOCDataset(
        dataset_dir='/home/wangcong/datasets/VOC/VOCdevkit/VOC2012',
        task='Detection',
        decode=True,
        num_parallel_workers=paralled_workers,
        cache=test_cache_1
        )
    else :
        dataset = mindspore.dataset.VOCDataset(
        dataset_dir='/home/wangcong/datasets/VOC/VOCdevkit/VOC2012',
        task='Detection',
        decode=True,
        num_parallel_workers=paralled_workers 
    )
       
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=1213414930, size=0, spilling=True)
        mapped_dataset = dataset.map(operations=rescale_op,cache=test_cache_2,num_parallel_workers=paralled_workers)
    else:
        mapped_dataset = dataset.map(operations=rescale_op,num_parallel_workers=paralled_workers)
    dataset = mapped_dataset.project(columns=["bbox", "label"])
    dataset = dataset.padded_batch(batch_size, drop_remainder=True,num_parallel_workers=paralled_workers,pad_info={"bbox": (None, -1),"label": (None, -1)})
    iterator=dataset.create_tuple_iterator(output_numpy=True)
    time_all=[]
    for epoch in range(10):
        start_time = time.time()
        for _ in iterator:
            data_nums = data_nums + 1
            # if data_nums > test_data_nums:
            #     break
        
        end_time = time.time()
        cost_time = end_time - start_time
        time_all.append(cost_time)

        print("mindspore cost time is: " + str(cost_time))
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}voc.csv")
    return cost_time, dataset

if __name__ == "__main__":
    bs=32
    num_worker=8
    # 测试各个
    #pytorch_imagefolder(bs, dataset_num_workers= num_worker)
    #tensorflow_imagefolder(bs, dataset_num_workers=num_worker)
    #tensorflow_imagefolder(bs, dataset_num_workers=8)
    # tensorflow_imagefolder(bs, dataset_num_workers=8)
    mindspore(bs, num_worker, 0,0)
    mindspore(bs, num_worker, 1,0)
    mindspore(bs, num_worker, 0,1)
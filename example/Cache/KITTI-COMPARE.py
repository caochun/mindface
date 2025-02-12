import os
import time
import numpy as np
import torch

import time
def pytorch_imagefolder(batch_size, dataset_num_workers=8):
    import torchvision
    data_nums = 0
    
    data_dir='/home/wangcong/data_unusual/'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.RandomAdjustSharpness(1),
        #torchvision.transforms.RandomVerticalFlip(),
        #torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    # 下载并加载训练集
    train_dataset = torchvision.datasets.Kitti(root=data_dir, train=True, download=False,
                                                transform=transform)
    

    
    def custom_collate_fn(batch):
        imgs, targets = zip(*batch)  # 分离图像和目标
        
        # 将图像堆叠成一个批次
        
        
        boxes = []
        labels = []

        for target in targets:
            
            b=[]
            l = []
            for obj in target:
                
                b.append(obj['bbox'])
                label_name = obj['type']
                l.append(label_name)

            boxes.append(np.array(b))
            labels.append(np.array(l))

        # 找到最大的 box 数量
        max_boxes = max(b.shape[0] for b in boxes)

        # 填充 boxes 和 labels
        padded_boxes = np.array([np.pad(b, ((0, max_boxes - b.shape[0]), (0, 0)), mode='constant', constant_values=-1) for b in boxes])
        padded_labels = np.array([np.pad(l, (0, max_boxes - l.shape[0]), mode='constant', constant_values=-1) for l in labels])

        return imgs, padded_boxes,padded_labels
    
    
    # 如果只想看前几个样本，可以使用 break
       
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                              num_workers=dataset_num_workers,collate_fn=custom_collate_fn)
    for epoch in range(2):
        start_time = time.time()
        for _ in train_loader:
            data_nums = data_nums + 1
            # if data_nums > test_data_nums:
            #     break
        print(data_nums)
        end_time = time.time()
        cost_time = end_time - start_time
        print("pytorch cost time is: " + str(cost_time))
    return cost_time



def tensorflow_imagefolder(batch_size, dataset_num_workers=8):
    import tensorflow as tf
    
    
   
    data_nums=0
    """
    def load_image(file_path):
        image = cv2.imread(file_path)
       
        return image
    """
    def load_kitti_data(data_dir, split='train', target_size=(256, 256)):
        image_paths = []
        bbox_annotations = []
        
        image_dir = os.path.join(data_dir, 'Kitti/raw/training/image_2') if split == 'train' else os.path.join(data_dir, 'Kitti/raw/testing/image_2')
        annotation_dir = os.path.join(data_dir, 'Kitti/raw/training/label_2') if split == 'train' else os.path.join(data_dir, 'Kitti/raw/testing/label_2')

        for filename in os.listdir(image_dir):
            
            image_paths.append(os.path.join(image_dir, filename))
            annotation_file = filename.replace('.png', '.txt')
            annotation_path = os.path.join(annotation_dir, annotation_file)
            
            
            with open(annotation_path, 'r') as f:
                bboxes = []
                for line in f:
                    parts = line.strip().split()
                    bbox = list(map(float, parts[4:8]))  # 这里只提取了 bbox 部分
                    bboxes.append(bbox)
                    
            bbox_annotations.append(bboxes)

        return image_paths, bbox_annotations
    def dataset_generator(image_paths, bbox_annotations):
        for img_path, bboxes in zip(image_paths, bbox_annotations):
            
            #print(img_path)
            yield img_path, bboxes
    data_dir = '/home/wangcong/data_unusual/'
    
    def preprocess_image(image, bboxes, target_size=(256, 256)):
        
        # 调整图像大小
        image_string = tf.io.read_file(image)
        image = tf.image.decode_png(image_string, channels=3)
        
        image = tf.image.resize(image, target_size)
    
       
        image = tf.transpose(image, [2, 1, 0])
        return image, bboxes
    
    image_paths, bbox_annotations = load_kitti_data(data_dir, split='train')
    
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(image_paths, bbox_annotations),
        output_types=(tf.string, tf.float32),
        output_shapes=(tf.TensorShape(None), tf.TensorShape([None, 4]))
    )
    dataset = dataset.map(preprocess_image, num_parallel_calls=dataset_num_workers)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([3, 256, 256], [None, 4]))
    #dataset = dataset.prefetch(buffer_size=dataset_num_workers * batch_size)

    for eoch in range(2):
        start_time = time.time()
        for batch in dataset:
            data_nums = data_nums + 1
            # 可以在这里进行处理，比如打印信息等

        end_time = time.time()
        cost_time = end_time - start_time
        print("tensorflow cost time is: " + str(cost_time))
    return cost_time




def mindspore(batch_size, num_parallel_workers=8, cache1=0,cache2=0):
    kitti_dataset_dir = "/home/wangcong/datasets/Kitti"
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore as ms
    
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
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
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()
    rank_id = get_rank()
    rank_size = get_group_size()
    data_nums=0
   
    
    
    rescale_op = vision.Resize([256,256])
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=2234436143, size=0, spilling=False)
        dataset = ds.KITTIDataset(shuffle=False,dataset_dir=kitti_dataset_dir, usage="train",decode=True,num_parallel_workers=num_parallel_workers,cache=test_cache_1
                                  ,num_shards=rank_size, shard_id=rank_id)
    
    else :
        dataset = ds.KITTIDataset(shuffle=False,dataset_dir=kitti_dataset_dir, usage="train",decode=True,num_parallel_workers=num_parallel_workers
                                  ,num_shards=rank_size, shard_id=rank_id)
       
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=2234436143, size=0, spilling=False)
    
        dataset = dataset.map(input_columns=["image"], operations=rescale_op, cache=test_cache_2,
                              num_parallel_workers=num_parallel_workers)
    
    else:
       dataset = dataset.map(input_columns=["image"], operations=rescale_op
                             ,num_parallel_workers=num_parallel_workers)
    

    dataset=dataset.project(['image', 'label','bbox'])
    dataset = dataset.padded_batch(batch_size, drop_remainder=True,num_parallel_workers=num_parallel_workers,pad_info={"bbox": (None, -1),"label": (None, -1)})
    #dataset = dataset.batch(batch_size)
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
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}Kitti.csv")
    return cost_time, dataset
if __name__ == "__main__":
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    bs=32
    num_workers=[8]
    for num_worker in num_workers:
        #tensorflow_imagefolder(bs,num_worker)
        #tensorflow_imagefolder(32,8)
        #pytorch_imagefolder(batch_size=bs,dataset_num_workers=num_worker)
        mindspore(bs, num_worker, 0,0)
        mindspore(bs, num_worker, 1,0)
        mindspore(bs, num_worker, 0,1)
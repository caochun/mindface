
import os
import time
from PIL import Image
import numpy as np



def pytorch_imagefolder(batch_size=32, paralled_workers=8):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    from torch.utils.data import DataLoader

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = load_dataset("csv", data_files={
    'train': '/home/wangcong/datasets/SST-2/train.tsv',
    
        }, delimiter='\t', cache_dir='/home/wangcong/datasets/')
    
    tokenizer = AutoTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')

    # 文本预处理函数
    def preprocess_data(example):
        inputs = tokenizer(example['sentence'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        label = example['label']  # 替换为你实际的标签列名
        return {
        'input_ids': inputs['input_ids'].squeeze(0), 
        'attention_mask': inputs['attention_mask'].squeeze(0), 
        'label': label
        }

# 处理数据集
    train_dataset = dataset['train'].map(preprocess_data)
    data_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
      # 使用自定义批处理函数
    num_workers=paralled_workers
            )
    data_nums = 0

    for epoch in range(2):
        start_time = time.time()
        for _ in data_loader:
            data_nums += 1
            # 如果有条件可以在此中断测试
            # if data_nums > test_data_nums:
            #     break

        end_time = time.time()
        cost_time = end_time - start_time

        print("PyTorch cost time is: " + str(cost_time))



def tensorflow_imagefolder(batch_size=32, parallel_workers=8):
    import numpy as np
    import tensorflow as tf
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    # 读取文本数据
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    dataset = load_dataset("csv", data_files={'train': '/home/wangcong/datasets/SST-2/train.tsv'}, delimiter='\t', cache_dir='/data1/datasets/')
    # 文本预处理函数
    def preprocess_data(sentence, label):
        inputs = tokenizer(sentence.numpy().decode('utf-8'), padding='max_length', truncation=True, max_length=128)
        return inputs['input_ids'], inputs['attention_mask'], label

    # 创建 TensorFlow 数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset['train']['sentence'], dataset['train']['label']))
    
    # 应用数据集映射
    train_dataset = train_dataset.map(lambda x, y: tf.py_function(preprocess_data, [x, y], [tf.int32, tf.int32, tf.int32]),
                                       num_parallel_calls=parallel_workers)


   

    # 设置批量大小和预取
    dataset = train_dataset.batch(batch_size)

    data_nums = 0
    for epoch in range(2):
        start_time = tf.timestamp()
        for _ in dataset:
            data_nums += 1
            # 如果有条件可以在此中断测试
            # if data_nums > test_data_nums:
            #     break

        end_time = tf.timestamp()
        cost_time = end_time - start_time

        print("TensorFlow cost time is: " + str(cost_time.numpy()))





def mindspore(batch_size, paralled_workers=8, cache1=0,cache2=0):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    data_nums=0
    from mindnlp.transformers import BertTokenizer
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
    
    tokenizer = BertTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    tokenizer.pad_token_id
    
    from transformers import AutoTokenizer
    #import mindspore.dataset.text as text
    
    #tokenizer = BertTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    
    #tokenizer = AutoTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    
    def tokenize_and_pad(text):
        #import pdb;pdb.set_trace()
        tokenized = tokenizer(text)

        return tokenized['input_ids'],tokenized['attention_mask']

    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=1058250972, size=0, spilling=False)
        dataset = ds.SST2Dataset(dataset_dir='/home/wangcong/datasets/SST-2',shuffle=False,cache=test_cache_1,num_parallel_workers=paralled_workers,num_shards=rank_size, shard_id=rank_id)

        
    else :
       dataset = ds.SST2Dataset(dataset_dir='/home/wangcong/datasets/SST-2',shuffle=False,num_parallel_workers=paralled_workers,num_shards=rank_size, shard_id=rank_id)

        
    
       
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=1058250972, size=0, spilling=False)
    
        dataset = dataset.map(operations=tokenize_and_pad,input_columns="sentence",output_columns=["input_ids","attention_mask"],cache=test_cache_2)
        

    else:
        dataset = dataset.map(operations=tokenize_and_pad,input_columns="sentence",output_columns=["input_ids","attention_mask"])

    dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id), 'attention_mask': (None, 0)})
    iterator=dataset.create_tuple_iterator(output_numpy=True)
    time_all=[]
    for epoch in range(8):
        start_time = time.time()
        for _ in iterator:
            data_nums = data_nums + 1
            # if data_nums > test_data_nums:
            #     break
        
        end_time = time.time()
        cost_time = end_time - start_time
        time_all.append(cost_time)

        print("mindspore cost time is: " + str(cost_time))
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}SST2.csv")
    return cost_time, dataset



if __name__ == "__main__":
    bs=32
    num_worker=8
    # 测试各个
    # test_ms_autotune()
    #pytorch_imagefolder(32,8)
    #tensorflow_imagefolder(32,8)
    mindspore(bs, num_worker, 0,0)
    mindspore(bs, num_worker, 1,0)
    mindspore(bs, num_worker, 0,1)
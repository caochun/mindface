import os
import time

def pytorch_imagefolder(batch_size=32, paralled_workers=8):  
    from transformers import AutoTokenizer
    from datasets import Dataset
    from torch.utils.data import DataLoader

    with open('/home/wangcong/datasets/wikitext-103/wiki.train.tokens', 'r', encoding='utf-8') as f:
        tokens = [line for line in f]
    
    dataset = Dataset.from_dict({"text": tokens})
    tokenizer = AutoTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')

    # 文本预处理函数
    def preprocess_data(example):
        inputs = tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)    
        return {'text': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}

    # 处理数据集
    train_dataset = dataset.map(preprocess_data)
    """
    def collate_fn(batch):
        input_ids =np.stack([item['text'] for item in batch])
        attention_mask = np.stack([item['attention_mask'] for item in batch])
        return  input_ids,attention_mask
    """
    data_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    
    num_workers=paralled_workers,
    #collate_fn=collate_fn
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
    
    with open('/home/wangcong/datasets/wikitext-103/wiki.train.tokens', 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f]
    # 创建 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    # 文本预处理函数
    def preprocess_data(example):
        inputs = tokenizer(example.numpy().decode('utf-8'), padding='max_length', truncation=True, max_length=128)
        return inputs['input_ids'], inputs['attention_mask']
    # 应用数据集映射
    dataset = dataset.map(lambda x: tf.py_function(preprocess_data, [x], [tf.int32, tf.int32]),
                          num_parallel_calls=parallel_workers)
    # 设置批量大小和预取
    dataset = dataset.batch(1)
    data_nums = 0
    
    for epoch in range(3):
        start_time = tf.timestamp()
        for _ in dataset:
            data_nums += 1
            # 如果有条件可以在此中断测试
            if data_nums > 200000:
                break

        end_time = tf.timestamp()
        cost_time = end_time - start_time
        print("TensorFlow cost time is: " + str(cost_time.numpy()))

def mindspore(batch_size, paralled_workers=8, cache1=0,cache2=0):
    
    import mindspore.dataset as ds
    #import mindspore.dataset.vision as vision
    data_nums=0
    from mindnlp.transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('/home/wangcong/data_unusual/bert')
    import mindspore.dataset.text as text
    vocab = text.Vocab.from_file("/home/wangcong/datasets/bert-base-uncased-vocab.txt")
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    import mindspore as ms
    
    from mindspore.communication import init
    from mindspore.communication import get_rank, get_group_size
    import csv
    
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
    
    def tokenize_and_pad(text):
        #import pdb;pdb.set_trace()
        tokenized = tokenizer(text.item(),truncation=True, 
                   max_length=128, padding='max_length')

        return tokenized['input_ids'],tokenized['attention_mask']
    
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=594859801, size=0, spilling=False)
        #dataset = mindspore.dataset.WikiTextDataset(dataset_dir='/home/wangcong/wikitext-103',shuffle=False,cache=test_cache_1,num_parallel_workers=paralled_workers)
        dataset = ds.WikiTextDataset(dataset_dir='/home/wangcong/wikitext-103',shuffle=False,num_parallel_workers=paralled_workers,cache=test_cache_1
                                     ,num_shards=rank_size, shard_id=rank_id) 
    else :
        #dataset = mindspore.dataset.WikiTextDataset(dataset_dir='/home/wangcong/wikitext-103',shuffle=False,num_parallel_workers=paralled_workers)
        dataset = ds.WikiTextDataset(dataset_dir='/home/wangcong/wikitext-103',shuffle=False,num_parallel_workers=paralled_workers
                                     ,num_shards=rank_size, shard_id=rank_id)
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=344190940, size=0, spilling=False)
        dataset = dataset.map(tokenizer_op, input_columns="text", num_parallel_workers=paralled_workers,cache=test_cache_2)  
    else:
        dataset = dataset.map(tokenizer_op, input_columns="text", num_parallel_workers=paralled_workers)
    #dataset = dataset.batch(1)

    #dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id), 'attention_mask': (None, 0)},num_parallel_workers=paralled_workers)
    #dataset = dataset.create_tuple_iterator(output_numpy=True)    
    dataset = dataset.padded_batch(batch_size, pad_info={'text': (None,'')},num_parallel_workers=paralled_workers)

    iterator=dataset.create_tuple_iterator(output_numpy=True)
    time_all=[]
    for epoch in range(5):
        data_nums=0
        start_time = time.time()
        for _ in iterator:
            data_nums = data_nums + 1
            
        end_time = time.time()
        cost_time = end_time - start_time
        time_all.append(cost_time)

        print("mindspore cost time is: " + str(cost_time))
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}wiki-new.csv")
    return cost_time, dataset

if __name__ == "__main__":
    bs=256
    num_worker=32
    #pytorch_imagefolder(bs,num_worker)
    #tensorflow_imagefolder(bs,num_worker)
    #mindspore(bs, num_worker, 0,0)
    # test_autotune_performance(10000)
    #mindspore(bs, num_worker, 1, 0)
    mindspore(bs, num_worker, 0,1)
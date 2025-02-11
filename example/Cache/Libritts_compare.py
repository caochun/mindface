import time

import numpy as np
import librosa

def pytorch_imagefolder(batch_size=32, paralled_workers=8):
    import torch
    from torch.utils.data import DataLoader
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
   
    def collate_fn_pad_sequence(batch):
        audios=[]
        texts=[]
        max_time_steps = 100
        # 提取音频和文本
        for item in batch:
            
            # 使用 scipy 的 signal.resample 进行重采样
            
            mfcc = np.transpose(librosa.feature.mfcc(y=np.array(item[0]), sr=item[1], n_mfcc=13).squeeze(0), (1,0))
            
        
            texts.append(item[2])
           
            if mfcc.shape[0] > max_time_steps:
                mfcc = mfcc[ :max_time_steps,:]  # 切片
            else:
                # 填充至 max_time_steps，使用零填充
                padding = np.zeros(( max_time_steps,13))
                padding[:mfcc.shape[0],: ] = mfcc
                mfcc = padding
            audios.append(mfcc)
        # 使用 pad_sequence 填充音频到相同长度
        
        
        return audios, texts

    # 加载 LibriTTS 数据集
    libritts_dataset = torchaudio.datasets.LIBRITTS(
        root='/home/wangcong/datasets',
        url='dev-clean',
        download=False
    )

    # 创建 DataLoader
    
    
    data_loader = DataLoader(
        libritts_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad_sequence,  # 使用自定义批处理函数
        num_workers=paralled_workers
    )
    data_nums = 0


    for epoch in range(3):
        start_time = time.time()
        for _ in data_loader:
            data_nums += 1
            # 如果有条件可以在此中断测试
            # if data_nums > test_data_nums:
            #     break

        end_time = time.time()
        cost_time = end_time - start_time

        print("PyTorch cost time is: " + str(cost_time))





def mindspore(batch_size, parallel_workers=8, cache1=0,cache2=0):
    import mindspore
    from mindspore import dataset as ds
    import numpy as np
    import time
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
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()
    rank_id = get_rank()
    rank_size = get_group_size()
    """

    
    # 加载 LibriTTS 数据集
    def pre_function(waveforms, sample_rate,original_text):   
        max_time_steps = 100
        
        
        
           
        resampled_waveform = librosa.feature.mfcc(y=waveforms, sr=sample_rate, n_mfcc=13)
        if resampled_waveform.shape[1] > max_time_steps:
            resampled_waveform = resampled_waveform[:, :max_time_steps]
        
        
        return resampled_waveform, sample_rate,original_text


    # 使用 DatasetCache 进行数据缓存（可选）
    
    if cache1:
        test_cache_1 = ds.DatasetCache(session_id=1822444736, size=0, spilling=False)
        libritts_dataset = ds.LibriTTSDataset(
       dataset_dir= "/home/wangcong/datasets/LibriTTS",
       usage='dev-clean',
       cache=test_cache_1 ,
       num_parallel_workers=parallel_workers,num_shards=rank_size, shard_id=rank_id
    )

        
    else :
        libritts_dataset = ds.LibriTTSDataset(
       dataset_dir= "/home/wangcong/datasets/LibriTTS",
       usage='dev-clean',
       
       num_parallel_workers=parallel_workers
    )

        
    
    #transforms = [audio.MFCC(4000, 128, 2)]  
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=1822444736, size=0, spilling=False)
    
        dataset = libritts_dataset.map(operations=pre_function,input_columns=['waveform', 'sample_rate','original_text'],output_columns=['waveform', 'sample_rate','original_text'],cache=test_cache_2)
        

    else:
        dataset = libritts_dataset.map(operations=pre_function,input_columns=['waveform', 'sample_rate','original_text'],output_columns=['waveform', 'sample_rate','original_text'])

 
    
        # 批量处理数据
    dataset = dataset.project(columns=['waveform', 'sample_rate','original_text'])
    dataset = dataset.padded_batch(batch_size, pad_info={'waveform': (None, 0.0)},num_parallel_workers=parallel_workers)
     
    """
    dataset =libritts_dataset.batch(batch_size, 
                                    per_batch_map=pre_function,
                                    input_columns=['waveform', 'original_text'],
                                    output_columns=["padded_wave", "padded_transcription"],
                                    num_parallel_workers=parallel_workers)
    """
    data_nums = 0
    time_all=[]
    for epoch in range(4):
        start_time = time.time()
        for _ in dataset:
            data_nums = data_nums + 1
            # if data_nums > test_data_nums:
            #     break
        
        end_time = time.time()
        cost_time = end_time - start_time
        time_all.append(cost_time)
        
        print(f"epoch{epoch}mindspore cost time is: " + str(cost_time))
    write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}Libritts.csv")

# 调用函数


def tensorflow_imagefolder(batch_size, num_parallel_workers):
    import time
    import os
    import numpy as np
    import tensorflow as tf
    
    
    max_time_steps = 100
    def generate_libritts(data_dir, subset):
        subset_path = os.path.join(data_dir, subset)  # 指定子数据集路径
        # 遍历指定子数据集目录下的所有文件
        print("enter__________________________________________________________________________________")
        for speaker_dir in os.listdir(subset_path):
            #print("speaker_path",speaker_dir)
            speaker_path = os.path.join(subset_path, speaker_dir)
            if os.path.isdir(speaker_path):
                for file_name in os.listdir(speaker_path):
                    #print("file",file_name)
                    final_path=os.path.join(speaker_path, file_name)
                    for last_name in os.listdir(final_path):
                        #print("lastname",last_name)
                        if last_name.endswith('.wav'):
                            # 读取音频文件
                            wafeform_path=os.path.join(final_path, last_name)
                            #waveform = tf.io.read_file(os.path.join(final_path, last_name))

                            #waveform, sample_rate = tf.audio.decode_wav(waveform)
                            #print("real?",waveform)
                            # 读取对应的文本文件（假设文本文件与音频文件同名）
                            transcription_file = os.path.splitext(last_name)[0] + '.original.txt'
                            with open(os.path.join(final_path, transcription_file), 'r') as f:
                                transcription = f.read().strip()
                            
                            yield wafeform_path, transcription # 返回音频和文本
    # 创建数据集
    subset='dev-clean'
    data_dir = '/home/wangcong/datasets/LibriTTS'  # LibriTTS 数据集路径


    def preprocess_image(wafeform_path, transcription):
        audio_path = wafeform_path.numpy().decode('utf-8')
        audio, sr = librosa.load(audio_path, sr=22050)  # 默认采样率 22050Hz
        mfcc = librosa.feature.mfcc(y=np.array(audio), sr=sr, n_mfcc=13)
        

        if mfcc.shape[1] > max_time_steps:
                mfcc = mfcc[:, :max_time_steps]  # 切片
        
        
        
        
        
        # 使用 pad_sequence 填充音频到相同长度
        #import pdb;
        #pdb.set_trace()
        
        return mfcc, transcription


        # 使用自定义生成器创建数据集
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_libritts(data_dir, subset),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string), # 音频
               
            tf.TensorSpec(shape=(), dtype=tf.string) # 文本
            
                          
        )
    )

    padded_shapes = ([13,None], [])  # 填充的形状 ([None] 表示填充音频长度, [] 表示文本不需要填充)
    padding_values = (0.0, '') 



    dataset = dataset.map(lambda x, y: tf.py_function(preprocess_image, [x, y],[tf.float32, tf.string]),
                                       num_parallel_calls=num_parallel_workers)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.map(preprocess_image, num_parallel_calls=num_parallel_workers)    
    # 数据集批处理，并设置填充规则
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

    # 数据集的预取
   

    # 测试数据加载和处理速度
    
    data_nums = 0
    start_time = time.time()
    for batch in dataset:
        data_nums += 1
        # 如果有条件可以在此中断测试
        # if data_nums > test_data_nums:
        #     break

    end_time = time.time()
    cost_time = end_time - start_time

    print("TensorFlow cost time is: " + str(cost_time))
    

if __name__ == "__main__":
    # 测试各个
    # test_ms_autotune()
    bs=32
    num_worker=8
    #pytorch_imagefolder(32,num_worker)
    tensorflow_imagefolder(32,8)
    #mindspore(bs, num_worker, 0,0)
    #mindspore(bs, num_worker, 1,0)
    #mindspore(bs, num_worker, 0,1)
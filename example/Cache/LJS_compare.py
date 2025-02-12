import os
import time

import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
import mindspore.dataset.audio as audio

import xml.etree.ElementTree as ET
import os
import time
from PIL import Image
import numpy as np
import math
import torchaudio


def pad_spectrogram(spec, max_length):
    current_length = spec.shape[-1]
    pad_width = max_length - current_length
    if pad_width > 0:
        padding = torch.zeros(1, 128, pad_width)  # 创建填充张量，保持与spec一致
        spec = torch.cat((spec, padding), dim=-1)
    return spec

def truncate_spectrogram(spec, max_length):
    # 截断谱图到指定的最大长度
    if spec.shape[-1] > max_length:
        spec = spec[:, :, :max_length]  # 截断
    return spec
def pytorch_imagefolder(batch_size=32, paralled_workers=8):
    n_fft = 1024
    win_length = None
    hop_length = 512
    import torchaudio
    import librosa
    from torch.utils.data import DataLoader, Dataset
    train_dataset =torchaudio.datasets.LJSPEECH(root='/home/wangcong/datasets', download=False)
    def pre_function(batch):
        audios_all = []
        text_all = []
        max_spec_length = 0

        for item in batch:
            audios, sr, text,text2 = item
            # 使用librosa进行带通滤波
            #audios = librosa.effects.preemphasis(audios)
            # 计算Spectrogram
            spectrogram_transform = librosa.feature.melspectrogram(
                y=audios.numpy(), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, power=2
            )
            # 转换为张量
            spectrogram_tensor = torch.tensor(spectrogram_transform)

            # 应用频率掩蔽
            freq_mask_param = 1
            if spectrogram_tensor.shape[1] > freq_mask_param:
                freq_mask = np.random.randint(0, freq_mask_param)
                spectrogram_tensor[:, freq_mask:] = 0  # 频率掩蔽

            # 更新最大长度
            #max_spec_length = max(max_spec_length, spectrogram_tensor.shape[1])
            audios_all.append(spectrogram_tensor)
            text_all.append(text)
        
            max_spec_length=256
        # 填充和截断
        audios_all = [truncate_spectrogram(aud, max_spec_length) for aud in audios_all]
        audios_all = [pad_spectrogram(aud, max_spec_length) for aud in audios_all]
        

        return audios_all, text_all, max_spec_length
    
    data_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=pre_function,  # 使用自定义批处理函数
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
        print(data_nums)
        print("PyTorch cost time is: " + str(cost_time))



def truncate_spectrogram(spec, max_length):
    # 截断谱图到指定的最大长度
    if spec.shape[-1] > max_length:
        spec = spec[:, :, :max_length]  # 截断
    return spec
def mindspore_imagefolder(batch_size=32, cache1=0,cache2=0,num_parallel_workers=8):
    import mindspore
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import librosa
    data_nums=0
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
    
    n_fft = 1024
    win_length = None
    hop_length = 512
    def pre_function(waveform, sample_rate, transcription):
            # 1. 带通滤波
        # 使用librosa的预加重函数，模拟 BandBiquad 的效果
        #waveform = librosa.effects.preemphasis(waveform)

        # 2. 计算频谱图
        
        spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0,
            center=True
        )

        # 3. 应用频率掩蔽
        freq_mask_param = 1
        if spectrogram.shape[0] > freq_mask_param:  # 频率维度
            freq_mask = np.random.randint(0, freq_mask_param)
            spectrogram[freq_mask:] = 0  # 频率掩蔽
        spectrogram=truncate_spectrogram(spectrogram,256)
        # 4. 处理文本数据
        texts = transcription

        return spectrogram, sample_rate, texts
        
        
    if cache1:
        test_cache = ds.DatasetCache(session_id=1822444736, size=0, spilling=False)
        dataset = ds.LJSpeechDataset(dataset_dir='/home/wangcong/datasets/LJSpeech-1.1',shuffle=False,cache=test_cache,num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = ds.LJSpeechDataset(dataset_dir='/home/wangcong/datasets/LJSpeech-1.1',shuffle=False,num_parallel_workers=num_parallel_workers,num_shards=rank_size, shard_id=rank_id)
  

    
    
    
    if cache2:
        test_cache_2 = ds.DatasetCache(session_id=1822444736, size=0, spilling=False)
        
        dataset = dataset.map(operations=pre_function, num_parallel_workers=num_parallel_workers,input_columns=['waveform', 'sample_rate', 'transcription']
                              ,output_columns=["padded_audios", 'sample_rate', 'transcription'],cache=test_cache_2)
    else:

        dataset = dataset.map(operations=pre_function, num_parallel_workers=num_parallel_workers,input_columns=['waveform', 'sample_rate', 'transcription']
                              ,output_columns=["padded_audios", 'sample_rate', 'transcription'])
    dataset=dataset.project(["padded_audios","sample_rate"])
    #dataset = dataset.batch(batch_size,num_parallel_workers=num_parallel_workers)
    dataset = dataset.padded_batch(batch_size, pad_info={"padded_audios": (None, 0)},num_parallel_workers=num_parallel_workers)
    dataset=dataset.create_tuple_iterator(output_numpy=True)
    
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
    #write_list_to_csv(time_all,f"/home/wangcong/performancetest/dietribued/{cache1}{cache2}LJS.csv")
    return cost_time, dataset


def tensorflow_imagefolder(batch_size=32, paralled_workers=8):
    import time
    import numpy as np
    import tensorflow as tf
    
    import librosa
    print("Eager execution enabled:", tf.executing_eagerly())
    n_fft = 1024
    win_length = None
    hop_length = 512

        
    
    class LJSpeechDataset(tf.data.Dataset):
        def __new__(cls, data_dir):
            # 读取文件名和路径
            metadata_file = os.path.join(data_dir, 'metadata.csv')
            with open(metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析元数据，获取音频路径和文本信息
            audio_paths = []
            
            for line in lines:
                parts = line.strip().split('|')
                audio_paths.append(os.path.join(data_dir, 'wavs', f"{parts[0]}.wav"))
                
            
            # 构造数据集
            dataset = tf.data.Dataset.from_tensor_slices(audio_paths)
            
            # 读取音频数据和文本
            
            
            # 映射数据集
            
            return dataset

    # 设置数据路径和参数
    data_dir = '/home/wangcong/datasets/LJSpeech-1.1' # LJSpeech 数据集路径
    def preprocess_data(audio_path):
        # 使用 librosa 加载音频文件
        audio_path = audio_path.numpy().decode('utf-8')
        audio, sr = librosa.load(audio_path, sr=22050)  # 默认采样率 22050Hz

        # 提取梅尔频谱
        melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, power=2)
        melspectrogram=np.squeeze(melspectrogram)
        melspectrogram=np.transpose(melspectrogram, (1,0))
        #print(melspectrogram.shape)
        # 返回梅尔频谱和文本
        return melspectrogram
  
    
    # 创建数据集
    dataset = LJSpeechDataset(data_dir)
    dataset = dataset.map(lambda x: tf.py_function(preprocess_data, [x], [tf.float32]), 
                          num_parallel_calls=paralled_workers)
    # 使用 padded_batch 自动填充音频数据到相同长度
    padded_shapes = ([128,None])  # 填充的形状 ([None] 表示填充音频长度, [] 表示文本不需要填充)
    padding_values = (0.0)   # 填充值 (音频填充为 0.0, 文本填充为空字符串)

    # 数据集批处理，并设置填充规则
    dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, 128]),))
    #dataset = dataset.batch(batch_size)
    # 数据集的预取
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 测试数据加载和处理速度
    start_time = time.time()
    data_nums = 0
    for data in dataset:
        data_nums += 1
        # 如果有条件可以在此中断测试
        # if data_nums > test_data_nums:
        #     break

    end_time = time.time()
    cost_time = end_time - start_time

    print("TensorFlow cost time is: " + str(cost_time))



if __name__ == "__main__":
    import tensorflow as tf
    
    
    # 测试各个
    # test_ms_autotune()
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    workers=[8]
    for worker in workers:
        print("current",worker)
        
        #pytorch_imagefolder(32,worker)
        tensorflow_imagefolder(32,worker)
        #mindspore_imagefolder(batch_size=32, cache1=0,cache2=0,num_parallel_workers=worker)  
        #mindspore_imagefolder(batch_size=32, cache1=1,cache2=0,num_parallel_workers=worker)  
        #mindspore_imagefolder(batch_size=32, cache1=0,cache2=1,num_parallel_workers=worker) 
        
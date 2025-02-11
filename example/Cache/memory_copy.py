import json
# run_monitor_process()
import argparse
import psutil
import time
import subprocess
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", default="32", type=str)
    parser.add_argument("--num_worker", default="8", type=str)
    
   
    parser.add_argument("--cpu_limit", default=False, type=bool)
    parser.add_argument("--cpu_num", default="30", type=str)
    parser.add_argument("--output", default="/home/wangcong/models/official/cv/SSD/cpu_output", type=str)
    
    
    args = parser.parse_args()
    return args

# python memory.py --type autotune --batch_size 32 --num_worker 16 --pipeline_rate 1 --optimize_path --output ./output/
def get_memory_info(process):
    return {
        "memory_full_info": process.memory_full_info(),
        # "memory_maps": process.memory_maps(),
        "memory_percent": process.memory_percent()
    }

def get_cpu_info(process):
    return {
        "cpu_num": process.cpu_num(),
        "cpu_percent": process.cpu_percent(),
        "cpu_times": process.cpu_times()
    }

def run_monitor_process():
    args = parse_args()
    if args.cpu_limit:
        print("cpu: limit true")
        process = subprocess.Popen([    "cpulimit", "-l", f"{args.cpu_num}", "--", 
                        'python', '/home/wangcong/mindcv/train.py', 
                        '-c', '/home/wangcong/mindcv/configs/resnet/resnet_50_gpu.yaml'
                    ])
        
    else:
        print("cpu: limit false")
        process = subprocess.Popen(['python', '/home/wangcong/models/official/cv/SSD/train.py', 
                        '--config', '/home/wangcong/models/official/cv/SSD/config/ssd_mobilenet_v1_300_config_gpu.yaml'])

    process_id = process.pid
    print("current process id is " + str(process_id))
    current_process = psutil.Process(process_id)
    output_name = "/cpulimit_" + str(args.cpu_limit) +  "_bs_" + str(args.batch_size) + "_worker_" + \
                    args.num_worker + "mobilev3_cache2"+"_output.csv"
    

    import os
    try:
        os.mkdir(args.output)
        print(1)
    except FileExistsError:
        print(args.output + " dir already exists")
    output_file = args.output +output_name 

    with open(output_file, 'w') as file:  # 打开文件以写入
        writer = csv.writer(file)
        writer.writerow(['cpu_num','cpu_times', 'cpu_percent', 'memory_full_info', 'memory_percent'])
        while process.poll() is None:
            cpu_info = get_cpu_info(current_process)
            memory_info = get_memory_info(current_process)

            info = [cpu_info['cpu_num'], cpu_info['cpu_times'], cpu_info['cpu_percent'], memory_info['memory_full_info'], memory_info['memory_percent']]
            # 写入文件
            # print("csv write one row data.....")
            writer.writerow(info)
            time.sleep(5)

run_monitor_process()

# python memory.py --type autotune --batch_size 32 --num_worker 16 --pipeline_rate 1 --iter_count 15 --optimize_path ./imagenet/ --output ./imagenet/
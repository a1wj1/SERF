import argparse
import json
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import math
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
from config import GlobalConfig
import time
from collections import deque
from thop import profile, clever_format
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from models.SERF_model import SERF

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from data.test_dataloader_ddd20 import DDD20_Data as EV
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

torch.cuda.empty_cache()
import warnings

warnings.filterwarnings("ignore")  # 过滤所有警告
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='Device to use')
#
parser.add_argument('--root_dir', type=str, default='/store/xxx/ddd20/dataset/ddd20', help='Path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--logdir', type=str, default='/datasets/xxx/SERF/log/SERF_ddd20_dark',
                    help='Directory to log data to.')
parser.add_argument('--num_workers', '-j', type=int, default=8)

args = parser.parse_args()
args.logdir = args.logdir


def count_parameters(model):
    """统计模型的可训练参数量（单位：百万）"""
    total = sum(p.numel() for p in model.parameters())  # 所有参数数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数
    return f"Total params: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M"


def get_model_size(model, dtype_size=4):
    """计算模型参数的内存占用（默认假设为float32，每个参数占4字节）"""
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = num_params * dtype_size
    size_mb = size_bytes / (1024 ** 2)  # 转换为MB
    return f"Memory: {size_mb:.2f} MB (assuming {dtype_size}-byte dtype)"


def print_model_summary(model, input_shape=(3, 224, 224)):
    # 生成虚拟输入
    dummy_input = torch.randn(1, *input_shape).to(next(model.parameters()).device)

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 打印结果
    print(f"Total params: {params / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f}G")  # 以 GigaFLOPs 为单位


def get_gpu_memory_usage(device_id=1):
    """获取指定GPU的显存使用情况"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # 返回MB

class Engine(object):
    """Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

    def __init__(self, config):
        self.config = config
        self.prediction = []
        self.prediction_plot = []
        self.actual = []
        self.actual_plot = []
        self.his_angle = []
        self.dvs_filename = []
        self.aps_filename = []
        self.input_buffer = {
            'his_angle': deque(maxlen=self.config.seq_len - 1),
        }
        self.step = 0

    def test(self):
        model.eval()
        # 正式计时
        total_time = 0.0
        total_samples = 0
        model_memory = []
        with torch.no_grad():
            rgb_input = torch.randn((1, 3, 256, 256))
            dvs_input = torch.randn((1, 3, 256, 256))
            dvs_image_frame = []
            aps_image_frame = []
            for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
                dvs_image_frame.append(dvs_input.to(args.device, dtype=torch.float32))
                aps_image_frame.append(rgb_input.to(args.device, dtype=torch.float32))

            MAC, params = profile(model, inputs=(dvs_image_frame, aps_image_frame), verbose=False)
            flops = 2 * MAC
            print(f"@1[INFO] FLOPs: {flops}, Params: {params}")

            flops, params = clever_format([flops, params], "%.3f")
            print(f"@2[INFO] FLOPs: {flops}, Params: {params}")

        with torch.no_grad():
            # Validation loop
            for batch_num, test_samples in enumerate(tqdm(dataloader_test)):
                if batch_num == 0:
                    continue
                torch.cuda.empty_cache()
                # 记录初始显存使用
                initial_memory = get_gpu_memory_usage()
                dvs_image_frame = []
                aps_image_frame = []
                for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
                    dvs_image_frame.append(test_samples['dvs_image_his_' + str(t)].to(args.device, dtype=torch.float32))
                    aps_image_frame.append(test_samples['aps_image_his_' + str(t)].to(args.device, dtype=torch.float32))

                angle = test_samples['gt_angle'].to(args.device, dtype=torch.float32)

                dvs_file = str(test_samples['dvs_filename'])[2:-2]
                aps_file = str(test_samples['aps_filename'])[2:-2]
                # print(dvs_file)
                self.dvs_filename.append(dvs_file)
                self.aps_filename.append(aps_file)

                start_time = time.perf_counter()

                pred_angle = model(dvs_image_frame, aps_image_frame)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                # 累计时间和样本数
                total_time += (end_time - start_time)
                total_samples += 1

                # 记录峰值显存使用
                peak_memory = get_gpu_memory_usage()
                # print("initial_memory:", initial_memory)
                # print("peak_memory:", peak_memory)

                # 计算模型显存占用
                model_memory.append(peak_memory - initial_memory)
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats()
                # 再次清空GPU缓存
                torch.cuda.empty_cache()
                gc.collect()

                # dvs_file = str(test_samples['dvs_filename'])[2:-2]
                # aps_file = str(test_samples['aps_filename'])[2:-2]
                #
                # self.dvs_filename.append(dvs_file)
                # self.aps_filename.append(aps_file)
                # self.his_angle.append(angle_frame)

                # self.prediction.append(str(math.degrees(pred_angle.cpu().numpy()[0][0])))
                # self.actual.append(str(math.degrees(angle.cpu().numpy()[0])))
                self.prediction.append(str(pred_angle.cpu().numpy()[0][0]))
                self.actual.append(str(angle.cpu().numpy()[0]))

                self.prediction_plot.append(pred_angle.cpu().numpy()[0][0])
                self.actual_plot.append(angle.cpu().numpy()[0])

            # 计算平均时间
            avg_time = total_time / total_samples
            print(f"Average inference time: {avg_time:.6f} seconds per sample")
            print(f"Total samples processed: {total_samples}")
            print(f"Average model memory: {np.mean(model_memory)}")

            print(count_parameters(model))
            print(get_model_size(model))
            # print_model_summary(model, input_shape=(3, 256, 256))  # 输入形状根据实际修改

    def save(self):
        result_dict = {'APS_filename': self.aps_filename, 'DVS_filename': self.dvs_filename, 'True_angle': self.actual,
                       'Predicted_angle': self.prediction}
        pandas_dataframe = pd.DataFrame(result_dict)
        pandas_dataframe.to_csv('res/SERF_ddd20_normal.csv')

    def plot(self):
        # np.savetxt('pred.csv',self.prediction)
        # np.savetxt('true_angle.csv',self.actual)
        MSE = mean_squared_error(self.actual_plot, self.prediction_plot)
        print('RMSE:', math.sqrt(MSE))
        MAE = mean_absolute_error(self.actual_plot, self.prediction_plot)
        print('MAE:', MAE)
        print(len(self.actual_plot))
        print(len(self.prediction_plot))
        # plt.plot(self.actual_plot, 'b', label="true")
        # plt.plot(self.prediction_plot, 'r', label="predicted")
        # plt.xlabel('num of images')
        # plt.ylabel('Angle')
        # plt.legend(loc="upper left")
        # plt.savefig('res/SERF_ddd20_normal.png')


if __name__ == '__main__':
    # Config
    config = GlobalConfig()
    csv_path = "/rec1487417411.csv"
    print(csv_path)
    # Data test_new_scene.csv  test_seq
    event_dataset = pd.read_csv(args.root_dir + csv_path)
    dataset_size = int(len(event_dataset))
    del event_dataset

    test_dataset = EV(root=args.root_dir, csv_file=csv_path, config=config)

    dataloader_test = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False)

    # model
    model = SERF(config)
    trainer = Engine(config)

    # 加载训练好的权重
    path_to_conf_file = os.path.join(args.logdir, 'best_epoch=129-val_loss=0.013.ckpt')
    ckpt = torch.load(path_to_conf_file)
    ckpt = ckpt["state_dict"]
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model.to(args.device)

    print('Loading checkpoint from ' + path_to_conf_file)

    trainer.test()
    trainer.plot()
    print(csv_path)
    # trainer.save()
    print('done')

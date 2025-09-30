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
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
from config import GlobalConfig
import random
from collections import deque

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from models.SERF_model import SERF

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from data.test_dataloader_carla import CARLA_Data as EV

torch.cuda.empty_cache()
import warnings

warnings.filterwarnings("ignore")  # 过滤所有警告
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:2', help='Device to use')
#
parser.add_argument('--root_dir', type=str, default='/home/xxxx/steer_test/carla/data', help='Path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--logdir', type=str,
                    default='/datasets/xxxx/SERF/log/SERF_carla',
                    help='Directory to log data to.')
parser.add_argument('--num_workers', '-j', type=int, default=8)

args = parser.parse_args()
args.logdir = args.logdir


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
        with torch.no_grad():
            # Validation loop
            for batch_num, test_samples in enumerate(tqdm(dataloader_test), 0):
                torch.cuda.empty_cache()
                dvs_image_frame = []
                aps_image_frame = []
                angle_frame = []
                for t in range(0, self.config.seq_len):  # 历史信息 + 当前信息
                    dvs_image_frame.append(test_samples['dvs_image_his_' + str(t)].to(args.device, dtype=torch.float32))
                    aps_image_frame.append(test_samples['aps_image_his_' + str(t)].to(args.device, dtype=torch.float32))

                angle = test_samples['gt_angle'].to(args.device, dtype=torch.float32)

                pred_angle = model(dvs_image_frame, aps_image_frame)

                self.input_buffer['his_angle'].append(pred_angle)

                dvs_file = str(test_samples['dvs_filename'])[2:-2]
                aps_file = str(test_samples['aps_filename'])[2:-2]

                self.dvs_filename.append(dvs_file)
                self.aps_filename.append(aps_file)
                # self.his_angle.append(angle_frame)

                # self.prediction.append(str(math.degrees(pred_angle.cpu().numpy()[0][0])))
                # self.actual.append(str(math.degrees(angle.cpu().numpy()[0])))
                self.prediction.append(str(pred_angle.cpu().numpy()[0][0]))
                self.actual.append(str(angle.cpu().numpy()[0]))

                self.prediction_plot.append(pred_angle.cpu().numpy()[0][0])
                self.actual_plot.append(angle.cpu().numpy()[0])

    def save(self):
        result_dict = {'APS_filename': self.aps_filename, 'DVS_filename': self.dvs_filename,
                       'True_angle': self.actual,
                       'Predicted_angle': self.prediction}
        pandas_dataframe = pd.DataFrame(result_dict)
        pandas_dataframe.to_csv('res/SERF_carla.csv')

    def plot(self):
        # np.savetxt('pred.csv',self.prediction)
        # np.savetxt('true_angle.csv',self.actual)
        MSE = mean_squared_error(self.actual_plot, self.prediction_plot)
        print('RMSE:', math.sqrt(MSE))
        MAE = mean_absolute_error(self.actual_plot, self.prediction_plot)
        print('MAE:', MAE)
        print(len(self.actual_plot))
        print(len(self.prediction_plot))
        plt.plot(self.actual_plot, 'b', label="true")
        plt.plot(self.prediction_plot, 'r', label="predicted")
        plt.xlabel('num of images')
        plt.ylabel('Angle')
        plt.legend(loc="upper left")
        plt.savefig('res/SERF_carla.png')


if __name__ == '__main__':
    # Config
    config = GlobalConfig()

    # Data test_new_scene.csv  test_seq
    event_dataset = pd.read_csv(args.root_dir + "/test_carla_normal.csv")
    dataset_size = int(len(event_dataset))
    del event_dataset

    test_dataset = EV(root=args.root_dir, csv_file="/test_carla_normal.csv", config=config)

    dataloader_test = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    # model
    model = SERF(config)
    trainer = Engine(config)

    # 加载训练好的权重
    path_to_conf_file = os.path.join(args.logdir, 'best_epoch=149-val_loss=0.006.ckpt')
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
    trainer.save()
    print('done')

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import math
from skimage.color import rgba2rgb
from skimage import io, transform
import pandas as pd


class CARLA_Data(Dataset):
    def __init__(self, root, csv_file, config, select_range=None, is_val=False, val_indices=None):
        """
        Args:
            root: 数据根目录
            csv_file: CSV文件路径
            config: 配置对象
            select_range: 用于指定数据范围的参数（保留原功能，但不用于随机采样）
            is_val: 是否为验证集（True表示验证集，False表示训练集）
            val_indices: 验证集的索引列表（仅当is_val=True时使用）
        """
        self.root = root
        self._batch_read_number = 0
        self.seq_len = config.seq_len

        camera_csv = pd.read_csv(self.root + csv_file)

        if is_val:
            if val_indices is None:
                raise ValueError("val_indices must be provided for validation dataset")
            self.camera_csv = camera_csv.iloc[val_indices].reset_index(drop=True)
        else:
            if val_indices is not None:
                all_indices = set(range(len(camera_csv)))
                train_indices = list(all_indices - set(val_indices))
                self.camera_csv = camera_csv.iloc[train_indices].reset_index(drop=True)
            else:
                self.camera_csv = camera_csv

        self.transform_enhance_rgb = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.transform_enhance_dvs = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.camera_csv)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()

        # his + cur frame
        frame_lens = len(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")) - self.seq_len
        for t, front_img in enumerate(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")):
            if t < len(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")) - self.seq_len:
                continue
            # print(t-frame_lens)
            key_name = 'aps_image_his_' + str(t-frame_lens)
            rgb_path = self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")[t][1:-1].replace(" ", "")
            rgb_path = rgb_path.replace("'", "")
            # print("rgb:", rgb_path)
            data[key_name] = self.transform_enhance_rgb(np.array(Image.open(self.root + rgb_path).convert('RGB')))

        for t, front_dvs in enumerate(self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")):
            if t < len(self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")) - self.seq_len:
                continue
            # print(t-frame_lens)
            key_name = 'dvs_image_his_' + str(t-frame_lens)
            dvs_path = self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")[t][1:-1].replace(" ", "")
            dvs_path = dvs_path.replace("'", "")
            # print("dvs:", dvs_path)
            data[key_name] = self.transform_enhance_dvs(np.array(Image.open(self.root + dvs_path).convert('RGB')))

        # cur
        fang_angle = self.camera_csv['True_angle'].iloc[index]
        data['gt_angle'] = float(fang_angle) # 方向盘预测

        self._batch_read_number += 1

        return data

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


class DDD20_Data(Dataset):
    def __init__(self, root, csv_file, config):
        self.root = root
        self._batch_read_number = 0
        camera_csv = pd.read_csv(self.root + csv_file)
        self.seq_len = config.seq_len

        self.camera_csv = camera_csv
        # image的transform
        self.transform_enhance_rgb = T.Compose([T.ToTensor(),
                                                T.Resize((256, 256)),
                                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                                ])
        self.transform_enhance_dvs = T.Compose([T.ToTensor(),
                                                T.Resize((256, 256)),
                                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                                ])

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.camera_csv)

    # {'aps_image_his_0': '/ddd20/rec1498946027_export/aps/201.png',
    #  'aps_image_his_1': '/ddd20/rec1498946027_export/aps/202.png',
    #  'aps_image_his_2': '/ddd20/rec1498946027_export/aps/203.png',
    #  'aps_image_his_3': '/ddd20/rec1498946027_export/aps/204.png',
    #  'aps_image_his_4': '/ddd20/rec1498946027_export/aps/205.png',
    #  'aps_image_his_5': '/ddd20/rec1498946027_export/aps/206.png',
    #  'aps_image_his_6': '/ddd20/rec1498946027_export/aps/207.png',
    #  'aps_image_his_7': '/ddd20/rec1498946027_export/aps/208.png',
    #  'dvs_image_his_0': '/ddd20/rec1498946027_export/dvs/201.png',
    #  'dvs_image_his_1': '/ddd20/rec1498946027_export/dvs/202.png',
    #  'dvs_image_his_2': '/ddd20/rec1498946027_export/dvs/203.png',
    #  'dvs_image_his_3': '/ddd20/rec1498946027_export/dvs/204.png',
    #  'dvs_image_his_4': '/ddd20/rec1498946027_export/dvs/205.png',
    #  'dvs_image_his_5': '/ddd20/rec1498946027_export/dvs/206.png',
    #  'dvs_image_his_6': '/ddd20/rec1498946027_export/dvs/207.png',
    #  'dvs_image_his_7': '/ddd20/rec1498946027_export/dvs/208.png'}
    # ------------------------------------

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        # data_test = dict()

        # his + cur frame
        frame_lens = len(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")) - self.seq_len
        for t, front_img in enumerate(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")):
            if t < len(self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")) - self.seq_len:
                continue
            # print(t-frame_lens)
            key_name = 'aps_image_his_' + str(t-frame_lens)
            rgb_path = self.camera_csv['APS_filename'].iloc[index][1:-1].split(",")[t][1:-1].replace(" ", "")
            rgb_path = rgb_path.replace("'", "")
            # print("rgb_key_name:", key_name)
            # print("rgb:", rgb_path)
            data[key_name] = self.transform_enhance_rgb(np.array(Image.open(self.root + rgb_path).convert('RGB')))
            # data_test[key_name] = rgb_path

        for t, front_dvs in enumerate(self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")):
            if t < len(self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")) - self.seq_len:
                continue
            # print(t-frame_lens)
            key_name = 'dvs_image_his_' + str(t-frame_lens)
            dvs_path = self.camera_csv['DVS_filename'].iloc[index][1:-1].split(",")[t][1:-1].replace(" ", "")
            dvs_path = dvs_path.replace("'", "")
            # print("dvs_key_name:", key_name)
            # print("dvs:", dvs_path)
            data[key_name] = self.transform_enhance_dvs(np.array(Image.open(self.root + dvs_path).convert('RGB')))
            # data_test[key_name] = dvs_path

        # print(data_test)
        # print("------------------------------------")

        # cur
        fang_angle = self.camera_csv['True_angle'].iloc[index]
        data['gt_angle'] = math.radians(float(fang_angle))  # 方向盘预测

        self._batch_read_number += 1

        return data

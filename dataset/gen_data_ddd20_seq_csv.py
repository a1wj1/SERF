from distutils.log import error
import os
import json
from typing import DefaultDict
import numpy as np
import tqdm
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

INPUT_FRAMES = 16
FUTURE_FRAMES = 1

de = [
    'rec1502336193',  #
    'rec1498946027',
    'rec1502338983',
    'rec1502339426',
]


def gen_single_route(route_folder):
    length = len(os.listdir(os.path.join(route_folder, 'measurements'))) - 1
    if length < INPUT_FRAMES + FUTURE_FRAMES:
        return

    full_steer = []

    seq_rgb_img = []
    seq_dvs_img = []
    his_steer = []
    cur_steer = []
    full_steer.append(0)

    for i in range(1, length):
        with open(os.path.join(route_folder, "measurements", f"{str(i)}.json"), "r", encoding='utf-8') as read_file:
            measurement = json.load(read_file)
            full_steer.append(measurement['steering_wheel_angle'])
    # print(full_steer)

    for i in range(INPUT_FRAMES - 1, length - FUTURE_FRAMES):
        # cur_steer.append(full_steer[i + 1: i + 1 + FUTURE_FRAMES])
        cur_steer.append(full_steer[i + 1])

        his_steer.append(full_steer[i - (INPUT_FRAMES - 1) + 1: i + 1])

        path = route_folder.replace(data_path, '')
        rgb_front_list = [path.replace('\\', '/') + "/aps/"f"{str(i - _ + 1)}.png" for _ in
                          range(INPUT_FRAMES - 1, -1, -1)]
        seq_rgb_img.append(rgb_front_list)

        dvs_front_list = [path.replace('\\', '/') + "/dvs/"f"{str(i - _ + 1)}.png" for _ in
                          range(INPUT_FRAMES - 1, -1, -1)]
        seq_dvs_img.append(dvs_front_list)

    # print(future_steer)
    return cur_steer, his_steer, seq_rgb_img, seq_dvs_img


def gen_sub_folder(folder_path):
    route_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
    route_list = sorted(route_list)

    total_rgb_img = []
    total_dvs_img = []
    total_steer = []
    total_future_steer = []

    for route in route_list:
        # print(route)
        if route not in de:
            continue
        print(route)
        seq_data = gen_single_route(os.path.join(folder_path, route))
        if not seq_data:
            continue
        cur_steer, _, seq_rgb_img, seq_dvs_img = seq_data

        result_dict_son = {'APS_filename': seq_rgb_img, 'DVS_filename': seq_dvs_img,
                           'True_angle': cur_steer}
        pandas_dataframe_son = pd.DataFrame(result_dict_son)

        # train_data, test_data = train_test_split(pandas_dataframe_son, test_size=0.1, random_state=42, shuffle=False)
        path_csv = os.path.join(folder_path, str(route) + '.csv')
        pandas_dataframe_son.to_csv(path_csv, index=False)

    return len(total_rgb_img)


if __name__ == '__main__':
    global data_path
    data_path = "/store/xxx/ddd20/dataset/ddd20"
    towns = ["ddd20"]
    pattern = "{}"  # town type
    import tqdm

    total = 0
    for town in tqdm.tqdm(towns):
        number = gen_sub_folder(os.path.join(data_path, pattern.format(town)))
        total += number

    print(total)

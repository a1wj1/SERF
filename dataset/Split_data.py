import os
from sklearn.model_selection import train_test_split
import pandas as pd

de = [
    'rec1502336193',  #
    'rec1498946027',
    'rec1502338983',
    'rec1502339426',
]

total_rgb_img = []
total_dvs_img = []
total_steer = []
total_future_steer = []

result_dict_train = {'APS_filename': total_rgb_img, 'DVS_filename': total_dvs_img, 'His_angle': total_steer,
                     'True_angle': total_future_steer}
pandas_dataframe_train = pd.DataFrame(result_dict_train)
result_dict_test = {'APS_filename': total_rgb_img, 'DVS_filename': total_dvs_img, 'His_angle': total_steer,
                    'True_angle': total_future_steer}
pandas_dataframe_test = pd.DataFrame(result_dict_test)
result_dict_val = {'APS_filename': total_rgb_img, 'DVS_filename': total_dvs_img, 'His_angle': total_steer,
                   'True_angle': total_future_steer}
pandas_dataframe_val = pd.DataFrame(result_dict_val)

# 指定文件夹路径
folder_path = 'C:/xxx/SERF/tools/ddd20'

# 遍历文件夹中的所有csv文件
for filename in os.listdir(folder_path):
    if filename.split('.')[0] not in de:
        continue
    print(filename)
    # 读取csv文件
    file_path = os.path.join(folder_path, filename)
    pandas_dataframe_son = pd.read_csv(file_path)

    test_data = pandas_dataframe_son.sample(frac=0.2, random_state=42)
    train_data = pandas_dataframe_son.drop(test_data.index)

    pandas_dataframe_train = pd.concat([pandas_dataframe_train, train_data])
    pandas_dataframe_test = pd.concat([pandas_dataframe_test, test_data])

pandas_dataframe_train.to_csv(folder_path + '/train_ddd20.csv', index=False)
pandas_dataframe_test.to_csv(folder_path + '/test_ddd20.csv', index=False)

print('完成')

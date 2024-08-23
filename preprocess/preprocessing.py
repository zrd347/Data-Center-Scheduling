import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler


def load_npy(target_file_path):
    if os.path.exists(target_file_path):
        loaded_data = np.load(target_file_path)
        return loaded_data
    else:
        print(f'{target_file_path} does not exist.')


class Preprocessing(object):
    def __init__(self, num_case, input_file_path, target_file_path):
        self.num_case = num_case
        self.input_file_path = input_file_path
        self.target_file_path = target_file_path
        self.input = []
        self.target = []
        self._preprocess_input()
        self._preprocess_target()

    def int_to_array(self, value):
        mapping = {
            0: [0, 0, 0, 0, 0],
            1: [1, 0, 0, 0, 0],
            2: [1, 1, 0, 0, 0],
            3: [1, 1, 1, 0, 0],
            4: [1, 1, 1, 1, 0],
            5: [1, 1, 1, 1, 1]
        }
        return mapping.get(value, [0, 0, 0, 0, 0])

    def _preprocess_input(self):
        df = pd.read_csv(self.input_file_path)
        for idx, row in df.iterrows():
            input_column = np.array([self.int_to_array(int(row[i])) for i in range(1, 1 + self.num_case)])
            self.input.append(input_column)
        # self.input = np.transpose(np.array(self.input), (1, 0))
        self.input = np.transpose(np.array(self.input), (1, 0, 2))
        self.input = self.input.reshape(self.num_case, 1000)



    def get_pos(self, target_data):
        pos_array = np.argwhere(np.array(target_data) == 1)
        pos = pos_array[0][0] * 45 + pos_array[0][1] + 1
        return pos

    def _preprocess_target(self):
        for i in range(self.num_case):
            target_file_path = f"{self.target_file_path}/result_{i+1:03}.npy"
            target_data = load_npy(target_file_path)
            count = 0
            target_row = []
            for j in range(200):
                for k in range(5):
                    if self.input[i][j * 5 + k] == 1:
                        target_row.append(self.get_pos(target_data[count]))
                        count += 1
                    else:
                        target_row.append(0)
            self.target.append(target_row)
        self.target = np.array(self.target)

    def creat_dataset(self, batch_size=32, train_ratio=0.8):
        X = torch.tensor(self.input, dtype=torch.float32)

        scaler = MinMaxScaler()
        self.target = scaler.fit_transform(self.target)

        y = torch.tensor(self.target, dtype=torch.float32)

        dataset = TensorDataset(X, y)

        # 定义训练集和测试集的大小
        train_size = int(train_ratio * len(dataset))  # 80% 作为训练集
        test_size = len(dataset) - train_size  # 剩余 20% 作为测试集

        # 使用 random_split 划分数据集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader



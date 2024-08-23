import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class Train:
    def __init__(self, model, train_loader, test_loader, num_epochs, lr, criterion):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = criterion
        # self.criterion = nn.MSELoss()

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0

            for inputs, targets in self.train_loader:
                # print(inputs[0])
                # print(targets[0])
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)
                # print(outputs[0])
                # 计算损失
                loss = self.criterion(outputs, targets)

                # 反向传播
                loss.backward()
                # 优化
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}')
        print('训练完成')

    def eval_model(self):
        self.model.eval()

        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                # outputs = torch.round(outputs)

                loss = self.criterion(outputs, targets)
                print(loss)

                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_outputs.append(outputs_np)
                all_targets.append(targets_np)

        all_outputs_np = np.concatenate(all_outputs, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)

        # 将 NumPy 数组转换为 DataFrame
        df_outputs = pd.DataFrame(all_outputs_np)
        df_targets = pd.DataFrame(all_targets_np)

        # 保存为 CSV 文件
        df_outputs.to_csv('output_100_cases/outputs_of_test_loader_100_cases.csv', index=False)
        df_targets.to_csv('output_100_cases/targets_of_test_loader_100_cases.csv', index=False)

        print("Files of acc compute saved to outputs.csv")


    def save_model(self):
        torch.save(self.model.state_dict(), 'model.pth')
        print('Model saved.')







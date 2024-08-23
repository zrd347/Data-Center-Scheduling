import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from preprocess.preprocessing import Preprocessing


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_case = 1000
input_file_path = "System Resource/input_for_transformer.csv"
target_file_path = "System Resource/New_Output"
data = Preprocessing(num_case, input_file_path, target_file_path)
train_loader, test_loader = data.creat_dataset()
print("finished data read")
# Initialize model, loss function, and optimizer
model = MLPModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
epochs = 400

# Training loop
train = []
for epoch in range(epochs):  # Number of epochs
    run_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

        batch_x_np = batch_x.cpu().numpy()
        train.append(batch_x_np)
    epoch_loss = run_loss / len(train_loader)
    print(f"Epoch:{epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}")


# Evaluate
model.eval()
all_outputs = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print(f"Test Loss:{loss}")

        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        df_outputs = pd.DataFrame(outputs_np)
        df_targets = pd.DataFrame(targets_np)

        df_outputs.to_csv('MLP_outputs_of_test_loader.csv', index=False)
        df_targets.to_csv('MLP_targets_of_test_loader.csv', index=False)

        # all_outputs.append(outputs_np)
        # all_targets.append(targets_np)

# all_outputs_np = np.concatenate(all_outputs, axis=0)
# all_targets_np = np.concatenate(all_targets, axis=0)
#
# # 将 NumPy 数组转换为 DataFrame
# df_outputs = pd.DataFrame(all_outputs_np)
# df_targets = pd.DataFrame(all_targets_np)
#
# # 保存为 CSV 文件
# df_outputs.to_csv('MLP_outputs_of_test_loader.csv', index=False)
# df_targets.to_csv('MLP_targets_of_test_loader.csv', index=False)

print("Files of MLP acc compute saved to outputs.csv")


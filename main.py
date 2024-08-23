import pandas as pd
import torch
from preprocess.preprocessing import Preprocessing
from model.loss import CustomLoss
from model.transformer import TransformerModel
from train.train import Train

# 创建 DataLoader
num_case = 100
input_file_path = "System Resource/input_for_transformer.csv"
target_file_path = "System Resource/New_Output"
data = Preprocessing(num_case, input_file_path, target_file_path)
# input = pd.DataFrame(data.input)
# input.to_csv("input_target/input.csv")
#
# target = pd.DataFrame(data.target)
# target.to_csv("input_target/target.csv")
# print(data.input[0])
# exit()
# print(data.input.shape)
# print(data.target.shape)
train_loader, test_loader = data.creat_dataset()


# 创建 Transformer Model
input_dim = 1000
output_dim = 1000
nhead = 8
num_layers = 6
model = TransformerModel(input_dim, output_dim, nhead, num_layers)
# model.load_state_dict(torch.load('model.pth'))

# model = TransformerModel(input_dim, output_dim, nhead, num_layers)

# 开始训练
num_epochs = 100
learning_rate = 0.000001
criterion = CustomLoss()

trainer = Train(model, train_loader, test_loader, num_epochs, learning_rate, criterion)
trainer.train_model()
trainer.save_model()
trainer.eval_model()


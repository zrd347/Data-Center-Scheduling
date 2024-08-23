import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(CustomLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

        # self.custom_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        # # 确保输出和目标的形状一致
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        # loss = self.custom_loss(outputs, targets)
        ce_loss = 0
        for i in range(len(outputs)):
            print(outputs[i], targets[i])
            ce_loss += nn.CrossEntropyLoss(weight=self.weight)(outputs[i], targets[i])
        ce_loss = torch.tensor(ce_loss, dtype=torch.int32)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss

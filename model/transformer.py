import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        # 定义Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward=512)
        # 将多个编码器层堆叠成Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输入的线性层（编码器）
        self.encoder = nn.Linear(input_dim, input_dim)
        # 输出的线性层（解码器）
        self.decoder = nn.Linear(input_dim, output_dim)
        # Sigmoid激活函数
        self.relu = nn.ReLU()

    def forward(self, src):
        # 初始输入形状为 (batch_size, sequence_length, input_dim)
        src = src.unsqueeze(2)
        src = src.permute(0, 2, 1)
        batch_size, sequence_length, input_dim = src.size()

        # 如果没有生成掩码或掩码的大小不等于序列长度，则生成一个新的掩码
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            device = src.device
            mask = self._generate_square_subsequent_mask(sequence_length).to(device)
            self.src_mask = mask

        # 对输入数据进行线性变换编码
        src = self.encoder(src)

        # 将输入形状调整为 (sequence_length, batch_size, input_dim) 以匹配Transformer编码器的要求
        src = src.permute(1, 0, 2)  # 调整形状
        # 通过Transformer编码器层处理输入数据
        output = self.transformer_encoder(src, self.src_mask)

        # 对编码器的输出进行线性变换解码
        output = self.decoder(output)
        # 添加激活函数
        output = self.relu(output)

        # 将输出形状调整回 (batch_size, sequence_length, output_dim)
        output = output.permute(1, 0, 2).contiguous()

        # 调整输出形状为 (batch_size, 200, input_dim, 2, 60)
        # output = output.view(batch_size, 200, input_dim, 2, 60)
        output = output.view(batch_size, 1000)
        # 通过激活函数处理输出
        output = self.relu(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        # 生成一个大小为 sz x sz 的方形掩码矩阵
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # 将掩码矩阵中上三角部分填充为0，其他部分填充为-inf
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

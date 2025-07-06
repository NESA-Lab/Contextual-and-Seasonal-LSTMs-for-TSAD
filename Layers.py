import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=nn.Tanh):
        super(MLPEncoder, self).__init__()
        modules = []
        in_channels = input_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, activation=nn.Tanh):
        super(MLP, self).__init__()
        modules = []
        in_channels = input_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation(),
                    nn.Dropout(p=dropout_rate)
                )
            )
            in_channels = h_dim
        
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1],input_dim+2)
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)
    

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.Tanh):
        super(MLPDecoder, self).__init__()
        # 解码器输入层
        self.decoder_input = nn.Linear(input_dim, hidden_dims[0])
        # 反转隐藏层维度顺序
        hidden_dims = hidden_dims[::-1]
        # 构建隐藏层
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    activation(),
                )
            )
        # 添加输出层
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim),
                activation(),
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        # 通过输入层
        x = self.decoder_input(z)
        # 调整shape
        x = x.view(-1, 1, 100)
        # 通过解码器层
        return self.decoder(x)


#卷积融合
class ConvFusion(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(ConvFusion, self).__init__()
        self.conv = nn.Conv1d(input_channels, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, time_seq, freq_seq):
        # 拼接通道 (batch, 2, length)
        x = torch.cat([time_seq, freq_seq], dim=1)
        # 卷积融合
        x = self.conv(x)  # (batch, 1, length)
        # x = x.squeeze(1)  # (batch, length)
        # 输出维度映射
        return self.fc(x)

#加权融合
class WeightedFusion(nn.Module):
    def __init__(self):
        super(WeightedFusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重

    def forward(self, time_seq, freq_seq):
        # 加权求和
        return self.alpha * time_seq + (1 - self.alpha) * freq_seq

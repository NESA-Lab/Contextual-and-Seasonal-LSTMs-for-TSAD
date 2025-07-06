import torch
import torch.nn as nn

class TCNLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation_rate):
        super(TCNLayer,self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=(kernel_size - 1) * dilation_rate // 2 if kernel_size%2==1 else (kernel_size - 1) * dilation_rate // 2+1,  # 确保长度不变
            bias=False
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self,x):
        out=self.conv(x)
        out=self.activation(out)
        out=self.dropout(out)
        return out

class TCN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,num_layers):
        super(TCN,self).__init__()
        layers=[]
        for i in range(num_layers):
            dilation_rate=2**i
            layers.append(
                TCNLayer(
                    in_channels=in_channels if i==0 else out_channels,
                    out_channels=1 if i==num_layers-1 else out_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate
                )
            )
        self.network=nn.Sequential(*layers)
        # self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1)
        self.fc = nn.Linear(1, 1)
        

    def forward(self,x):
        tcn_out=self.network(x)
        # x_a=x.permute(2, 0, 1)  
        # attn_out, _ = self.attention(x_a, x_a, x_a)  
        # attn_out = attn_out.permute(1, 2, 0)
        # attn_out = self.fc(attn_out.permute(0,2,1))
        # attn_out = attn_out.permute(0, 2, 1)
        return tcn_out
    
import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
import pywt

class AnomalyTransformer(nn.Module):
    """
    初始化模型参数
    :param input_dim: 每个时间步的输入维度（channel）
    :param d_model: Transformer 的嵌入维度
    :param nhead: 多头注意力的头数
    :param num_layers: Transformer 编码器层的层数
    :param num_classes: 分类类别数
    :param dropout: Dropout 概率
    """
    def __init__(
        self,
        hp,
    ):
        super(AnomalyTransformer,self).__init__()
        self.hp=hp

        hour_encoder_layer=nn.TransformerEncoderLayer(d_model = 64,
                                                 dim_feedforward = 64,
                                                 nhead = 4,
                                                 dropout=self.hp.dropout_rate)  
        
        #hour encoder
        self.hour_embedding = nn.Linear(8, 64)
        self.hour_encoder = nn.TransformerEncoder(hour_encoder_layer, 
                                                 num_layers=self.hp.num_layers)
        #生成掩码
        src_mask = self.generate_causal_mask(24).to(self.hp.gpu)
        self.src_mask = src_mask.masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, 0)

        self.hour_position_embedding = nn.Parameter(torch.zeros(24, 600, 64))
        self.hour_linear = nn.Sequential(
            nn.Linear(64, 8),
            nn.Tanh()
        )
        

        day_encoder_layer=nn.TransformerEncoderLayer(d_model=self.hp.d_model,
                                                 dim_feedforward = self.hp.d_model,
                                                 nhead=self.hp.n_head,
                                                 dropout=self.hp.dropout_rate)  

        #day encoder
        self.day_embedding = nn.Linear(2*self.hp.input_dim + 2, self.hp.d_model)
        self.num = int((self.hp.window - self.hp.input_dim)//24 + 1 )
        self.day_position_embedding = nn.Parameter(torch.zeros(self.num ,600, self.hp.d_model))
        self.day_encoder = nn.TransformerEncoder(day_encoder_layer,
                                                 num_layers=self.hp.num_layers)
        
        self.day_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        #参数融合，进行预测
        self.predict_layer = nn.Linear(self.num+1, 1)

    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内时序性
        hour_input = input.clone()
        hour_input = hour_input.unfold(dimension=2,size = self.hp.input_dim, step=24)
        hour_input = hour_input.squeeze(1)
        # 调整shape为 （batch，hour，day）
        hour_input = hour_input.permute(0,2,1)


        #整个窗口进行切分，求天与天之间周期性
        day_input = input.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.input_dim, step=24)
        day_input = day_input.squeeze(1)
        #预测的那一天设置为0，不参与预测
        f_day_input = day_input.clone()
        #添加频率信息
        f_day_input= torch.fft.rfft(f_day_input,dim=-1)
        f_day_input=torch.cat((f_day_input.real,f_day_input.imag),dim=-1)
        #将时序信息与频率信息按channel拼接
        day_input = torch.cat((day_input, f_day_input),dim=-1)
        # 调整shape为 （batch，hour，day）
        day_input = day_input.permute(0,2,1)

        return day_input,hour_input
    
    def encode(self,input):
        day_input, hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour attention, 调整为 （hour, batch, day)
        hour_input = hour_input.permute(1,0,2)
        hour_input = self.hour_embedding(hour_input) + self.hour_position_embedding[:,:batch,:]
        hour_output = self.hour_encoder(hour_input, mask = self.src_mask)
        hour_output = self.hour_linear(hour_output)
        # 计算 day attention, 调整为 （day, batch, hour）
        day_input = day_input.permute(2,0,1)
        day_input = self.day_embedding(day_input) + self.day_position_embedding[:,:batch,:]
        day_output = self.day_encoder(day_input)
        day_output = self.day_linear(day_output)

        #只保留窗口中的最后一个元素, 调整shape为（batch，1，window=channel）
        #取最后一个小时(batch,day)
        hour_output = hour_output[-1,:,:]
        #取计算周期性后的最后一小时(batch,1)
        day_output = day_output[-1,:,-1].unsqueeze(1)
        #特征融合
        result = torch.cat((hour_output,day_output),dim=-1)
        result = self.predict_layer(result)
        return result


    def forward(self,input, input_normal, mode):
        """
        前向传播
        :param input: 输入张量，shape = (batch, 1, window)
        """
        if mode == "train" or mode == "valid":
            loss = self.loss_func(input, input_normal)
            return loss
        else:
            return self.MCMC2(input)

    def loss_func(self, input, input_normal):
        #通过之前的点进行预测
        result = self.encode(input)
        #获得真实值
        ground_truth = input_normal[:,:,-1]

        loss = torch.mean(
            torch.mean((ground_truth - result)**2, dim=1),
            dim=0
        )

        if torch.isinf(loss):
            raise
        return loss
        
    
    def MCMC2(self,input):
        result = self.encode(input)
        input = input[:,:,-1]
        loss = (result - input)**2
        return loss
    
    def generate_causal_mask(self,seq_len):
        # 生成下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask
        

    
            



    


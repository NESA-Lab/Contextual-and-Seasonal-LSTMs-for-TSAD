import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from cycle_model import CycleVAE
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
        self.hp = hp
        self.cycleVae = CycleVAE(hp)
        self.lstm = nn.LSTM(2*self.hp.input_dim + 2, self.hp.d_model, batch_first = True)
        self.cycle_lstm = nn.LSTM(2*self.hp.input_dim + 2, self.hp.d_model, batch_first = True)
        
        self.hour_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        self.day_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        #重建层,通道融合
        self.hour_mu = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        self.hour_log_var = nn.Linear(self.hp.input_dim, self.hp.input_dim)

        self.day_mu = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        self.day_log_var = nn.Linear(self.hp.input_dim, self.hp.input_dim)


    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = input.clone()
        # hour_input = hour_input[:,:,-self.hp.input_dim*2 + 1:]
        hour_input = hour_input[:,:,-self.hp.input_dim - 7:]
        hour_input = hour_input.unfold(dimension=2,size = self.hp.input_dim,step=1)
        hour_input = hour_input.squeeze(1)
        # hour_input = hour_input[:,:-1,:]
        #预测的那天设置为0，不参与预测
        f_hour_input = hour_input.clone()
        # #添加频率信息
        f_hour_input = torch.fft.rfft(f_hour_input,dim=-1)
        hour_input = torch.cat((hour_input, f_hour_input.real, f_hour_input.imag),dim=-1)
        # print(hour_input.dtype)  # 打印第一级低频系数的数据类型

        #整个窗口进行切分，求天与天之间的关系
        day_input = input.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.input_dim, step=self.hp.input_dim)
        day_input = day_input.squeeze(1)
        # day_input = day_input[:,:-1,:]
        f_day_input = day_input.clone()
        f_day_input = torch.fft.rfft(f_day_input, dim=-1)
        day_input = torch.cat([day_input, f_day_input.real, f_day_input.imag],dim=-1)

        return day_input,hour_input
    
    def encode(self,input):
        day_input,hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour result
        result, (h,c) = self.lstm(hour_input)
        result = self.hour_linear(result[:,-1,:]).unsqueeze(1)
        hour_mu = self.hour_mu(result)
        hour_log_var = self.hour_log_var(result)
        # 计算 day result
        # 修改为 （batch，day，hour）
        # mu_x, log_var_x, kl_loss = self.cycleVae(day_input)
        result, (h,c) = self.cycle_lstm(day_input)
        result = self.day_linear(result[:,-1,:]).unsqueeze(1)
        day_mu = self.day_mu(result)
        day_log_var = self.day_log_var(result)

        #特征融合
        return day_mu, day_log_var, hour_mu, hour_log_var


    def forward(self, input, input_normal, mode, mask):
        """
        前向传播
        :param input: 输入张量，shape = (batch, 1, window)
        """
        if mode == "train" or mode == "valid":
            loss = self.loss_func(input, input_normal, mask)
            return loss
        else:
            return self.MCMC2(input)

    def loss_func(self, input, input_normal, mask):
        day_mu, day_log_var, hour_mu, hour_log_var = self.encode(input)
        input_normal = input_normal[:,:,-self.hp.input_dim:].squeeze(1)
        mask = mask[:,-self.hp.input_dim:]
        hour_mu = hour_mu.squeeze(1)
        hour_log_var = hour_log_var.squeeze(1)
        day_mu = day_mu.squeeze(1)
        day_log_var = day_log_var.squeeze(1)

        num = torch.sum(mask,dim=-1)


        hour_recon_loss = torch.mean(
            0.5*torch.sum((hour_log_var + (input_normal - hour_mu)**2/torch.exp(hour_log_var))*mask,dim=-1)/(num+1e-9),
            dim = 0
        )

        cycle_recon_loss = torch.mean(
            0.5*torch.sum((day_log_var + (input_normal - day_mu)**2/torch.exp(day_log_var))*mask,dim=-1)/(num+1e-9),
            dim = 0
        )

        loss = cycle_recon_loss + hour_recon_loss

        if torch.isinf(loss):
            raise
        return loss
        
    
    def MCMC2(self,input):
        loss = 0
        input_copy = input.clone()
        input = input[:,:,-self.hp.input_dim:]
        day_mu, day_log_var, hour_mu, hour_log_var = self.encode(input_copy)
        loss += (hour_log_var + (input - hour_mu)**2/torch.exp(hour_log_var))*0.5 + (day_log_var + (input - day_mu)**2/torch.exp(day_log_var))*0.5
        return loss
    
    
            



    


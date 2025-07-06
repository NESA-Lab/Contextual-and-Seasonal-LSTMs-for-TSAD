import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
import pywt
import math
from torchmetrics.utilities.data import dim_zero_mean


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
        self.lstm = nn.LSTM(2*self.hp.stride + 2, self.hp.d_model//4, batch_first = True, dropout = self.hp.dropout_rate)
        self.cycle_lstm = nn.LSTM(2*self.hp.input_dim + 2, self.hp.d_model, batch_first = True, dropout = self.hp.dropout_rate)


        self.hour_linear = nn.Sequential(
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.stride + 2),
            nn.Tanh()
        )

        self.day_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.input_dim + 2),
            nn.Tanh()
        )

        self.hour_mu = nn.Linear(self.hp.stride, self.hp.stride)
        self.hour_log_var = nn.Linear(self.hp.stride, self.hp.stride)

        self.day_mu = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        self.day_log_var = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        


    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = input.clone()
        hour_input = hour_input[:,:,-self.hp.stride*6:]
        hour_input = hour_input.unfold(dimension=2,size = self.hp.stride,step = self.hp.step)
        hour_input = hour_input.squeeze(1)
        #预测的那天设置为0，不参与预测
        # hour_input = hour_input[:,:-1,:]
        f_hour_input = hour_input.clone() 
        f_hour_input[:,-1,-1]=0
        # #添加频率信息
        f_hour_input = self.get_freq(f_hour_input)
        hour_input = torch.cat((hour_input, f_hour_input),dim=-1)

        #整个窗口进行切分，求天与天之间的关系
        day_input = input.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.input_dim, step=self.hp.cycle)
        day_input = day_input.squeeze(1)
        # day_input[:,-1,-1]=0
        f_day_input = day_input.clone()
        f_day_input[:,-1,-1]=0
        f_day_input = self.get_freq(f_day_input)
        day_input = torch.cat((day_input, f_day_input),dim=-1)

        return day_input,hour_input
        # return  day_input, hour_input
        # return  f_day_input, f_hour_input
    
    def encode(self,input):
        day_input,hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour result
        result, h = self.lstm(hour_input)

        result_hour = self.hour_linear(result[:,-1,:]).unsqueeze(1)
        hour_real = result_hour[:, :, :(self.hp.stride // 2 + 1)]
        hour_imag = result_hour[:, :, :(self.hp.stride // 2 + 1)]
        f_hour = torch.stack((hour_real,hour_imag),dim=-1)
        f_hour = torch.view_as_complex(f_hour)
        result_hour = torch.fft.irfft(f_hour)
        hour_mu = self.hour_mu(result_hour)
        hour_log_var = self.hour_log_var(result_hour)

        # 计算 day result
        # 修改为 （batch，day，hour）
        # mu_x, log_var_x, kl_loss = self.cycleVae(day_input)
        result, h = self.cycle_lstm(day_input)

        result_day = self.day_linear(result[:,-1,:]).unsqueeze(1)
        day_real = result_day[:,:,:(self.hp.input_dim//2 + 1)]
        day_imag = result_day[:,:,(self.hp.input_dim//2 + 1):]
        f_day = torch.stack((day_real,day_imag),dim=-1)
        f_day = torch.view_as_complex(f_day)
        result_day = torch.fft.irfft(f_day)
        day_mu = self.day_mu(result_day)
        day_log_var = self.day_log_var(result_day)


        #特征融合
        return day_mu, day_log_var, hour_mu, hour_log_var

    def forward(self, input, input_normal, mode, mask):
        # mask_copy = mask.clone()
        # mask_copy[:,-1] = 1
        # mask_copy = mask_copy.unsqueeze(1)
        # input = input*mask_copy
        """
        前向传播
        :param input: 输入张量，shape = (batch, 1, window)
        """
        if mode == "train" or mode == "valid":
            loss = self.loss_func(input, input_normal, mask)
            return loss
        else:
            return self.MCMC2(input, input_normal)

    def loss_func(self, input, input_normal, mask):
        day_mu, day_log_var, hour_mu, hour_log_var = self.encode(input)
        # mu, var = self.encode(input)
        input_day = input_normal[:,:,-self.hp.input_dim:].squeeze(1)
        input_hour = input_normal[:,:,-self.hp.stride:].squeeze(1)
        mask_day = mask[:, -self.hp.input_dim:]
        mask_hour = mask[:, -self.hp.stride:]
        hour_mu = hour_mu.squeeze(1)
        hour_log_var = hour_log_var.squeeze(1)
        day_mu = day_mu.squeeze(1)
        day_log_var = day_log_var.squeeze(1)


        num_day = torch.sum(mask_day,dim=-1)
        num_hour = torch.sum(mask_hour,dim=-1)



        hour_recon_loss = torch.mean(
            0.5*torch.sum((hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var))*mask_hour,dim=-1)/(num_hour + 1e-9),
            dim = 0
        )



        hour_ub = hour_mu + torch.sqrt(torch.exp(hour_log_var))
        hour_lb = hour_mu - torch.sqrt(torch.exp(hour_log_var))
        hour_filter_loss = torch.mean(
            torch.mean((torch.max(torch.zeros_like(input_hour), -input_hour + hour_ub) + torch.max(torch.zeros_like(input_hour), -hour_lb + input_hour))*torch.logical_not(mask_hour) +
                       (torch.max(torch.zeros_like(input_hour), input_hour - hour_ub) + torch.max(torch.zeros_like(input_hour), hour_lb - input_hour))*mask_hour,dim=-1),
            dim=0
        )

        day_recon_loss = torch.mean(
            0.5*torch.sum((day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))*mask_day,dim=-1)/(num_day + 1e-9),
            dim = 0
        )


        day_ub = day_mu + torch.sqrt(torch.exp(day_log_var))
        day_lb = day_mu - torch.sqrt(torch.exp(day_log_var))
        day_filter_loss = torch.mean(
            torch.mean((torch.max(torch.zeros_like(input_day), -input_day + day_ub) + torch.max(torch.zeros_like(input_day), -day_lb + input_day))*torch.logical_not(mask_day)+
                       (torch.max(torch.zeros_like(input_day), input_day - day_ub) + torch.max(torch.zeros_like(input_day), day_lb - input_day))*mask_day,dim=-1),
            dim=0
        )


        loss = hour_recon_loss + day_recon_loss + day_filter_loss + hour_filter_loss
        if torch.isinf(loss):
            raise
        return loss
        
    
    def MCMC2(self,input, input_normal):
        loss = 0
        input_day = input_normal[:, :,-self.hp.input_dim:]
        input_hour = input_normal[:, :, -self.hp.stride:]
        day_mu, day_log_var, hour_mu, hour_log_var = self.encode(input)

        hour_loss = (hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var))
        hour_loss = hour_loss[:,:,-1].unsqueeze(2)

        cycle_loss = (day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))
        cycle_loss = cycle_loss[:,:,-1].unsqueeze(2)

        day_ub = day_mu + 3*torch.sqrt(torch.exp(day_log_var))
        day_lb = day_mu - 3*torch.sqrt(torch.exp(day_log_var))
        day_ub = day_ub[:,:,-1].unsqueeze(2)
        day_lb = day_lb[:,:,-1].unsqueeze(2)

        hour_ub = hour_mu + 3*torch.sqrt(torch.exp(hour_log_var))
        hour_lb = hour_mu - 3*torch.sqrt(torch.exp(hour_log_var))
        hour_ub = hour_ub[:,:,-1].unsqueeze(2)
        hour_lb = hour_lb[:,:,-1].unsqueeze(2)

        # ub = torch.min(day_ub, hour_ub)
        # lb = torch.max(day_lb, hour_lb)
        # ub = ub[:,:,-1]
        # lb = lb[:,:,-1]

        loss =  hour_loss + cycle_loss

        recon_x = (hour_mu[:,:,-1].unsqueeze(2) + day_mu[:,:,-1].unsqueeze(2))/2
        # day_log_var = day_log_var[:,:,-1].unsqueeze(2)
        # hour_log_var = hour_log_var[:,:,-1].unsqueeze(2)
        # ub = recon_x + torch.min(torch.sqrt(torch.exp(day_log_var)),torch.sqrt(torch.exp(hour_log_var)))
        # lb = recon_x - torch.min(torch.sqrt(torch.exp(day_log_var)),torch.sqrt(torch.exp(hour_log_var)))
        return loss, recon_x, day_ub, day_lb, hour_ub, hour_lb

    
    def get_freq(self, input):
        f_global = torch.fft.rfft(input, dim = -1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        return f_global



    
    
            



    


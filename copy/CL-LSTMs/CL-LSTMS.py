import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
import pywt
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
        self.lstm = nn.LSTM(2*self.hp.stride + 2, self.hp.d_model//4, batch_first = True, num_layers=1, dropout = self.hp.dropout_rate)
        self.cycle_lstm = nn.LSTM(2*self.hp.input_dim + 2, self.hp.d_model, batch_first = True, num_layers=1, dropout = self.hp.dropout_rate)
        
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

        # self.confusion_layer = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1),
        #     nn.Tanh()
        # )

        # self.mu_layer = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        # self.var_layer = nn.Linear(self.hp.input_dim, self.hp.input_dim)


    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = input.clone()
        hour_input = hour_input[:,:,-self.hp.stride*4:]
        hour_input = hour_input.unfold(dimension=2,size = self.hp.stride,step = 4)
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
    
    def encode(self,input):
        day_input,hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour result
        result, (h,c) = self.lstm(hour_input)
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
        result, (h,c) = self.cycle_lstm(day_input)
        result_day = self.day_linear(result[:,-1,:]).unsqueeze(1)
        day_real = result_day[:,:,:(self.hp.input_dim//2 + 1)]
        day_imag = result_day[:,:,(self.hp.input_dim//2 + 1):]
        f_day = torch.stack((day_real,day_imag),dim=-1)
        f_day = torch.view_as_complex(f_day)
        result_day = torch.fft.irfft(f_day)
        day_mu = self.day_mu(result_day)
        day_log_var = self.day_log_var(result_day)

        # confusion_result = self.confusion_layer(torch.cat((result_hour, result_day),dim=1))
        # mu = self.mu_layer(confusion_result)
        # var = self.var_layer(confusion_result)

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
        # moving_day = self.get_move_mean(input_normal[:,:,-self.hp.input_dim-5:])
        # moving_hour = self.get_move_mean(input_normal[:,:,-self.hp.stride-5:])
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

        cycle_recon_loss = torch.mean(
            0.5*torch.sum((day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))*mask_day,dim=-1)/(num_day + 1e-9),
            dim = 0
        )

        # hour_recon_loss = torch.mean(
        #     torch.mean(((hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var))*mask_hour
        #                + (hour_log_var + (moving_hour - hour_mu)**2/torch.exp(hour_log_var))*torch.logical_not(mask_hour)),
        #               dim=-1),
        #     dim = 0
        # )
        
        # cycle_recon_loss = torch.mean(
        #     torch.mean(((day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))*mask_day
        #                + (day_log_var + (moving_day - day_mu)**2/torch.exp(day_log_var))*torch.logical_not(mask_day)),
        #               dim=-1),
        #     dim = 0
        # )

        loss = hour_recon_loss + cycle_recon_loss


        # loss = torch.mean(
        #     0.5*torch.sum((var + (input_normal - mu)**2/torch.exp(var))*mask,dim=-1)/(num+1e-9),
        #     dim = 0
        # )

        if torch.isinf(loss):
            raise
        return loss
        
    
    def MCMC2(self,input, input_normal):
        loss = 0
        input_day = input_normal[:, :,-self.hp.input_dim:]
        input_hour = input_normal[:, :, -self.hp.stride:]
        day_mu, day_log_var, hour_mu, hour_log_var = self.encode(input)
        # mu, var = self.encode(input_copy)
        hour_loss = (hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var))
        hour_loss = hour_loss[:,:,-1].unsqueeze(2)

        cycle_loss = (day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))
        cycle_loss = cycle_loss[:,:,-1].unsqueeze(2)

        loss +=  hour_loss + cycle_loss
        recon_x = (hour_mu[:,:,-1] + day_mu[:,:,-1]).unsqueeze(2)
        return loss, recon_x

    
    def get_freq(self, input):
        f_global = torch.fft.rfft(input, dim = -1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        # num = self.hp.input_dim//self.hp.stride
        # f_local = []
        # for i in range(num):
        #     f_temp = torch.fft.rfft(input[:,:,(i*self.hp.stride):((i+1)*self.hp.stride)])
        #     f_local.append(torch.cat((f_temp.real, f_temp.imag),dim=-1))
        # f_local = torch.cat(f_local,dim=-1)
        # f_local = self.local_emb(f_local)
        return f_global

    def get_move_mean(self, data):
        move_size = 5
        windows = data.unfold(dimension=-1, size= move_size, step=1)
        moving_average = torch.mean(windows, dim=-1).squeeze(1)
        moving_average = moving_average[:,:-1]
        return moving_average




    
    
            



    


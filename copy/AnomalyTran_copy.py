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

        hour_encoder_layer=nn.TransformerEncoderLayer(d_model= self.hp.d_model,
                                                 dim_feedforward = self.hp.d_model,
                                                 nhead=self.hp.n_head,
                                                 dropout=self.hp.dropout_rate)  
        
        #hour encoder
        self.hour_embedding = nn.Linear(2*self.hp.input_dim+2, self.hp.d_model)
        # self.hour_position_embedding = nn.Parameter(torch.zeros(24 ,600, self.hp.d_model))
        self.hour_encoder = nn.TransformerEncoder(hour_encoder_layer, 
                                                 num_layers=self.hp.num_layers)
        
        self.hour_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        day_encoder_layer=nn.TransformerEncoderLayer(d_model=self.hp.d_model,
                                                 dim_feedforward = self.hp.d_model,
                                                 nhead=self.hp.n_head,
                                                 dropout=self.hp.dropout_rate)  

        #day encoder
        self.day_embedding = nn.Linear(2*self.hp.input_dim + 2, self.hp.d_model)
        self.num = int((self.hp.window - self.hp.input_dim)/24 + 1 )
        # self.day_position_embedding = nn.Parameter(torch.zeros(num,600, self.hp.d_model))
        self.day_encoder = nn.TransformerEncoder(day_encoder_layer,
                                                 num_layers=self.hp.num_layers)
        
        self.day_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        self.position_embedding = nn.Parameter(torch.zeros(24 ,600, self.hp.d_model))
        #重建层,通道融合
        self.confusion_layer = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=1)
        self.fc_mu = nn.Linear(self.hp.input_dim,self.hp.input_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.hp.input_dim,self.hp.input_dim),
            nn.Softplus()
        )

    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = input.clone()
        hour_input = hour_input[:,:,-self.hp.input_dim - 23:]
        hour_input = hour_input.unfold(dimension=2,size = self.hp.input_dim,step=1)
        hour_input = hour_input.squeeze(1)
        #预测的那天设置为0，不参与预测
        f_hour_input = hour_input.clone()
        f_hour_input[:,-1,-1]=0
        # #添加频率信息
        f_hour_input= torch.fft.rfft(f_hour_input,dim=-1)
        f_hour_input=torch.cat((f_hour_input.real,f_hour_input.imag),dim=-1)
        #将时序信息与频率信息按channel拼接
        hour_input = torch.cat((hour_input, f_hour_input),dim=-1)
        # 调整shape为 （batch，channel，window）
        hour_input = hour_input.permute(0,2,1)

        #整个窗口进行切分，求天与天之间的关系
        day_input = input.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.input_dim, step=24)
        day_input = day_input.squeeze(1)
        #预测的那一天设置为0，不参与预测
        f_day_input = day_input.clone()
        f_day_input[:,-1,-1]=0
        #添加频率信息
        f_day_input= torch.fft.rfft(f_day_input,dim=-1)
        f_day_input=torch.cat((f_day_input.real,f_day_input.imag),dim=-1)
        #将时序信息与频率信息按channel拼接
        day_input = torch.cat((day_input, f_day_input),dim=-1)
        # 调整shape为 （batch，channel，window）
        day_input = day_input.permute(0,2,1)

        return day_input,hour_input
    
    def encode(self,input):
        day_input,hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour attention
        hour_input = hour_input.permute(2,0,1)
        hour_input = self.hour_embedding(hour_input) + self.position_embedding[:,:batch,:]
        hour_output = self.hour_encoder(hour_input)
        hour_output = self.hour_linear(hour_output)
        #计算 day attention
        day_input = day_input.permute(2,0,1)
        day_input = self.day_embedding(day_input) + self.position_embedding[:self.num,:batch,:]
        day_output = self.day_encoder(day_input)
        day_output = self.day_linear(day_output)

        # print(torch.isnan(day_output).any())
        #只保留窗口中的最后一个元素, 调整shape为（batch，1，window=channel）
        hour_output = hour_output[-1,:,:].unsqueeze(0).permute(1,0,2)
        day_output = day_output[-1,:,:].unsqueeze(0).permute(1,0,2)
        #特征融合
        result = torch.cat((hour_output,day_output),dim=1)
        result = self.confusion_layer(result)
        # result = day_output
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return mu,var


    def forward(self,input, input_normal, mode, mask):
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
        # result = self.encode(input)
        mu,var = self.encode(input)
        input_normal = input_normal[:,:,-self.hp.input_dim:].squeeze(1)
        # result = result.squeeze(1)
        mu = mu.squeeze(1)
        var = var.squeeze(1)
        # mask_normal = mask[:,-self.hp.input_dim:].squeeze(1)
        # num_normal = torch.sum(mask_normal,dim=-1)
        # mask_abnormal = torch.logical_not(mask_normal)
        # num_abnormal = torch.sum(mask_abnormal,dim=-1)
        # mu = mu.squeeze(1)
        # var = var.squeeze(1)
        
        # mask=mask[:,-self.hp.input_dim:]

        #正态损失
        loss = torch.mean(
            0.5
            * torch.mean(torch.log(var) + (input_normal - mu) ** 2 / var, dim=1),
            dim=0,
        )

        # # 均方损失
        # m = torch.sum(mask, dim=1, keepdim=True).repeat(1,self.hp.input_dim)
        # mu = (torch.sum(
        #     input*mask,
        #     dim = -1,
        #     keepdim = True
        # ).repeat(1,self.hp.input_dim))/(m + 1e-9)

        # mu = mu*mask + input

        # loss = torch.mean(
        #     torch.mean(torch.abs(result-mu),dim=-1),
        #     dim=0
        # )
        # loss = torch.mean(
        #     torch.mean((input_normal - result)**2, dim=1),
        #     dim=0
        # )

        # # 正常均方误差
        # loss = torch.mean(
        #     #只求正常点的重构误差
        #     torch.sum(mask_normal*(input_normal - result)**2,dim=-1)/(num_normal+1e-9) + torch.sum(mask_abnormal*(input_normal - result)**2,dim=-1)/(num_abnormal+1e-9),
        #     dim=0
        # )
        if torch.isinf(loss):
            raise
        return loss
        
    # def forward(self,input, mode, mask):
    #     """
    #     前向传播
    #     :param input: 输入张量，shape = (batch, 1, window)
    #     """
    #     if mode == "train" or mode == "valid":
    #         loss = self.loss_func(input, mask)
    #         return loss
    #     else:
    #         return self.MCMC2(input)

    # def loss_func(self, input, mask):
    #     result = self.encode(input)
    #     input = input[:,:,-self.hp.input_dim:].squeeze(1)
    #     result = result.squeeze(1)
    #     mask = mask[:,-self.hp.input_dim:].squeeze(1)

    #     loss = torch.mean(
    #         torch.mean(mask*(input - result)**2, dim=1),
    #         dim=0
    #     )

    #     # # 正常均方误差
    #     # loss = torch.mean(
    #     #     #只求正常点的重构误差
    #     #     torch.mean(mask_normal*(input_normal - result)**2+mask_abnormal*(input_normal - result)**2*5, dim=1),
    #     #     dim=0
    #     # )
    #     if torch.isinf(loss):
    #         raise
    #     return loss
    
    def MCMC2(self,input):
        # result = self.encode(input)
        mu, var =self.encode(input)
        input = input[:,:,-self.hp.input_dim:]
        # loss = (result - input)**2
        loss = torch.log(var) + (input - mu) ** 2 / var
        return loss
        

    
            



    


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

        hour_encoder_layer=nn.TransformerEncoderLayer(d_model= self.hp.d_model,
                                                 dim_feedforward = self.hp.d_model,
                                                 nhead=self.hp.n_head,
                                                 dropout=self.hp.dropout_rate)  
        #生成掩码
        src_mask = self.generate_causal_mask(self.hp.input_dim).to(self.hp.gpu)
        self.src_mask = src_mask.masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, 0)
        
        #hour encoder
        self.hour_embedding = nn.Linear(2*self.hp.input_dim + 2, self.hp.d_model)
        self.hour_encoder = nn.TransformerEncoder(hour_encoder_layer, 
                                                 num_layers=self.hp.num_layers)
        
        self.hour_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.input_dim),
            nn.Tanh()
        )

        self.position_embedding = nn.Parameter(torch.zeros(self.hp.input_dim ,600, self.hp.d_model))
        #重建层,通道融合
        self.hour_mu = nn.Linear(self.hp.input_dim, self.hp.input_dim)
        self.hour_log_var = nn.Linear(self.hp.input_dim, self.hp.input_dim)


    def get_input(self, input):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = input.clone()
        hour_input = hour_input[:,:,-self.hp.input_dim*2 + 1:]
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
        # 调整shape为 （batch，hour，day）
        
        # #提取趋势信息
        # trends = hour_input.clone()
        # trends[:,-1,-1]=0
        # trends = self.extract_wavelet_trend(trends)
        # hour_input = torch.cat((hour_input,trends),dim=-1)
        hour_input = hour_input.permute(0,2,1)
        # print(hour_input.dtype)  # 打印第一级低频系数的数据类型

        #整个窗口进行切分，求天与天之间的关系
        day_input = input.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.input_dim, step=self.hp.input_dim)
        day_input = day_input.squeeze(1)
        # 调整shape为 （batch，hour，day）
        day_input = day_input.permute(0,2,1)

        return day_input,hour_input
    
    def encode(self,input):
        day_input,hour_input = self.get_input(input)
        batch,channel,window = input.shape
        
        # 计算 hour result
        hour_input = hour_input.permute(2,0,1)
        hour_input = self.hour_embedding(hour_input) + self.position_embedding[:,:batch,:]
        hour_output = self.hour_encoder(hour_input, mask = self.src_mask)
        hour_output = self.hour_linear(hour_output)
        hour_output = hour_output[-1,:,:].unsqueeze(0).permute(1,0,2)
        hour_mu = self.hour_mu(hour_output)
        hour_log_var = self.hour_log_var(hour_output)

        # 计算 day result
        # 修改为 （batch，day，hour）
        day_input = day_input.permute(0,2,1)
        mu_x, log_var_x, kl_loss = self.cycleVae(day_input)

        #特征融合
        return mu_x, log_var_x, kl_loss, hour_mu, hour_log_var


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
        mu_x, log_var_x, kl_loss, hour_mu, hour_log_var = self.encode(input)
        input_normal = input_normal[:,:,-self.hp.input_dim:].squeeze(1)
        hour_mu = hour_mu.squeeze(1)
        hour_log_var = hour_log_var.squeeze(1)
        mu_x = mu_x.squeeze(1)
        log_var_x = log_var_x.squeeze(1)

        hour_recon_loss = torch.mean(
            0.5*torch.mean(hour_log_var + (input_normal - hour_mu)**2/torch.exp(hour_log_var),dim=-1),
            dim = 0
        )

        cycle_recon_loss = torch.mean(
            0.5*torch.mean(log_var_x + (input_normal - mu_x)**2/torch.exp(log_var_x),dim=-1),
            dim = 0
        )

        loss =  kl_loss + cycle_recon_loss + hour_recon_loss

        if torch.isinf(loss):
            raise
        return loss
        
    
    def MCMC2(self,input):
        loss = 0
        input_copy = input.clone()
        input = input[:,:,-self.hp.input_dim:]
        for i in range(10):
            mu_x, log_var_x, kl_loss, hour_mu, hour_log_var = self.encode(input_copy)
            # loss = (result - input)**2
            loss += (hour_log_var + (input - hour_mu)**2/torch.exp(hour_log_var))*0.5 + (log_var_x + (input - mu_x)**2/torch.exp(log_var_x))*0.5
        loss /= 10
        return loss
    
    def generate_causal_mask(self,seq_len):
        # 生成下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask
    
    
            



    


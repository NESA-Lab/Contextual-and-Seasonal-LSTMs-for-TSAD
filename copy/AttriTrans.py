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
        # Embedding
        self.input_embedding = nn.Linear(self.hp.input_dim, self.hp.d_model)
        self.positional_encoding = PositionalEncoding(self.hp.d_model,self.hp.dropout_rate).to(self.hp.gpu)
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hp.d_model,
                                                        dim_feedforward=self.hp.d_model,
                                                        nhead=self.hp.n_head,
                                                        dropout=self.hp.dropout_rate)
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=self.hp.num_layers)
        
        self.output_layer = nn.Linear(self.hp.d_model,1)
    
    def forward(self,input,mode):
        batch,window,channel = input.shape
        src = input[:,:-1,:]
        ground_truth = input[:,-1,:]
    
        src = self.input_embedding(src)
        #添加位置编码
        src = self.positional_encoding(src).permute(1,0,2)
        output = self.encoder(src)
        output = output[-1,:,:]
        output = self.output_layer(output)
        #将 shape 还原为 (batch,window,channel)

        if mode == "train" or mode == "valid":
            loss = self.loss_func(output,ground_truth)
            return loss
        else:
            return self.MCMC2(output,ground_truth)

    
    def loss_func(self,output,ground_truth):
        output = output.squeeze(1)
        # ground_truth = ground_truth.squeeze(1)
        ground_truth = ground_truth[:,0]
       
        loss = torch.mean(
            (output-ground_truth)**2,
            dim=0
        )
        return loss

    def MCMC2(self,output,ground_truth):
        output = output.squeeze(1)
        # ground_truth = ground_truth.squeeze(1)
        ground_truth = ground_truth[:,0]
        loss = (output-ground_truth)**2
        return loss


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #计算位置编码
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        #shape 修改为(batch,window,channel)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.shape[1],:]
        return self.dropout(x)

    
            



    


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
from Layers import MLP
import pywt

class FMLP(nn.Module):
    def __init__(
            self,
            hp,
            max_capacity: int = 25,
            Capacity_max_iter: int = 1e5,
            loss_type: str = "C"
    ):
        super(FMLP,self).__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.hidden_dims = [100,100,100,100]

        #编码器
        self.mlp = MLP(input_dim=self.hp.window, hidden_dims=self.hidden_dims,dropout_rate=self.hp.dropout_rate)


    def forward(self,input,mode,mask):
        if mode == "train" or mode == "valid":
            f_input = torch.fft.rfft(input[:,:,:-1], dim=-1)
            f_input = torch.cat((f_input.real,f_input.imag),dim=-1)
            result = self.mlp(f_input)
            loss = self.loss_func(result, input, mask)
            return loss
        else:
            return self.MCMC2(input)

    def loss_func(self, mu_x, input, mask):
        input = input.squeeze(1)
        mu_x = mu_x.squeeze(1)

        # # 线性插值
        interpolated_series = self.linear_interpolation(input,mask)
        f_inter = torch.fft.rfft(interpolated_series,dim=-1)
        f_inter = torch.cat((f_inter.real,f_inter.imag),dim=-1)
        loss = torch.mean(
            torch.mean((f_inter-mu_x)**2,dim=-1),
            dim=0
        )

        #掩盖掉异常点，保留正常的频率信息
        # input = input*mask
        # recon_loss = 0

        # for i in range(len(input)):
        #     input_i = input[i]
        #     input_i = input_i[torch.where(input_i!=0)]
        #     f_input_i = torch.fft.rfft(input_i)
        #     mu_x_i = mu_x[i]
        #     l = len(f_input_i)
        #     mu_x_i_real = mu_x_i[:l]
        #     mu_x_i_imag = mu_x_i[(len(mu_x_i)//2-1):(len(mu_x_i)//2-1+l)]
        #     mu_x_i = torch.cat((mu_x_i_real,mu_x_i_imag))
        #     f_input_i = torch.cat((f_input_i.real,f_input_i.imag))
        #     recon_loss += torch.mean(
        #         (mu_x_i-f_input_i)**2
        #     )
        # loss = recon_loss/input.shape[0]

        # m = torch.sum(mask,dim=-1)

        # #计算误差
        # loss = torch.mean(
        #     torch.sum(mask*(input-t_mu_x)**2,dim=-1)/m,
        #     dim=0,
        # )
        return loss
        
    def MCMC2(self,input):
        f_input=torch.fft.rfft(input[:,:,:-1],dim=-1)
        f_input=torch.cat((f_input.real,f_input.imag),dim=-1)
        result = self.mlp(f_input)
        l = result.shape[-1]
        result = torch.stack((result[:,:,:(l//2)],result[:,:,(l//2):]),dim=-1)
        result = torch.view_as_complex(result)
        t_result = torch.fft.irfft(result,dim=-1)
        loss_all = (input-t_result)**2
        #只保留最后一个点的误差
        loss_all = loss_all[:,:,-1]
        return loss_all


    def linear_interpolation(self, input, mask):
        """
        对异常点进行线性插值
        :param input: 输入的时间序列，形状为 (batch, window)
        :param mask: 标记正常点和异常点的掩码，形状为 (batch, window)，1 表示正常点，0 表示异常点
        :return: 插值后的时间序列
        """
        # 确保输入和掩码的形状一致
        assert input.shape == mask.shape
        
        # 获取 batch 和 window 的大小
        batch_size, window_size = input.shape
        
        # 复制输入序列作为输出
        output = input.clone()
        
        for i in range(batch_size):
            for j in range(window_size):
                if mask[i, j] == 0:  # 如果是异常点
                    # 找到左边最近的正常点
                    left_idx = j - 1
                    while left_idx >= 0 and mask[i, left_idx] == 0:
                        left_idx -= 1
                    # 找到右边最近的正常点
                    right_idx = j + 1
                    while right_idx < window_size and mask[i, right_idx] == 0:
                        right_idx += 1
                    
                    # 如果左右都有正常点，则进行线性插值
                    if left_idx >= 0 and right_idx < window_size:
                        left_value = input[i, left_idx]
                        right_value = input[i, right_idx]
                        # 线性插值公式
                        output[i, j] = left_value + (right_value - left_value) * (j - left_idx) / (right_idx - left_idx)
                    elif left_idx >= 0:  # 只有左边有正常点，右边是异常点
                        output[i, j] = input[i, left_idx]
                    elif right_idx < window_size:  # 只有右边有正常点，左边是异常点
                        output[i, j] = input[i, right_idx]
        
        return output



import torch
import numpy as np
from torch import nn
from time_series_decomp import Series_decop


class CSLSTMs(nn.Module):
    def __init__(
        self,
        hp,
    ):
        super(CSLSTMs,self).__init__()
        self.hp = hp
        self.lstm = nn.LSTM(2*self.hp.contextual_window + 2, self.hp.d_model//4, batch_first = True, dropout = self.hp.dropout_rate)
        self.cycle_lstm = nn.LSTM(2*self.hp.seasonal_window + 2, self.hp.d_model, batch_first = True, dropout = self.hp.dropout_rate)
        self.series_decop = Series_decop(level = int(np.round(np.log2(self.hp.window))-4), hp=self.hp)

        self.hour_linear = nn.Sequential(
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.d_model//4),
            nn.Tanh(),
            nn.Linear(self.hp.d_model//4, self.hp.contextual_window + 2),
            nn.Tanh()
        )

        self.day_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.d_model),
            nn.Tanh(),
            nn.Linear(self.hp.d_model, self.hp.seasonal_window + 2),
            nn.Tanh()
        )

        self.hour_mu = nn.Linear(self.hp.contextual_window, self.hp.contextual_window)
        self.hour_log_var = nn.Linear(self.hp.contextual_window, self.hp.contextual_window)

        self.day_mu = nn.Linear(self.hp.seasonal_window, self.hp.seasonal_window)
        self.day_log_var = nn.Linear(self.hp.seasonal_window, self.hp.seasonal_window)
        


    def get_input(self, trend_signal):
        """
        将窗口进一步切分，用于后续送入attention进行处理
        """
        #取最后24个元素，求一天内的影响
        hour_input = trend_signal.clone()
        hour_input = hour_input[:,:,-self.hp.contextual_window*4:]
        hour_input = hour_input.unfold(dimension=2,size = self.hp.contextual_window,step = self.hp.step)
        hour_input = hour_input.squeeze(1)
        #预测的那天设置为0，不参与预测
        # hour_input[:,-1,-1]=0
        f_hour_input = hour_input.clone() 
        f_hour_input[:,-1,-1]=0
        # #添加频率信息
        f_hour_input = self.get_freq(f_hour_input)
        hour_input = torch.cat((hour_input, f_hour_input),dim=-1)

        #整个窗口进行切分，求天与天之间的关系
        day_input = trend_signal.clone()
        day_input = day_input.unfold(dimension=2,size = self.hp.seasonal_window, step=self.hp.cycle)
        day_input = day_input.squeeze(1)
        # day_input[:,-1,-1]=0
        f_day_input = day_input.clone()
        f_day_input[:,-1,-1]=0
        f_day_input = self.get_freq(f_day_input)
        day_input = torch.cat((day_input, f_day_input),dim=-1)



        return day_input,hour_input
        # return  day_input, hour_input
        # return  f_day_input, f_hour_input
    
    def encode(self,trend_signal):
        day_input,hour_input = self.get_input(trend_signal)
        
        # 计算 hour result
        result, h = self.lstm(hour_input)

        result_hour = self.hour_linear(result[:,-1,:]).unsqueeze(1)
        hour_real = result_hour[:, :, :(self.hp.contextual_window // 2 + 1)]
        hour_imag = result_hour[:, :, (self.hp.contextual_window // 2 + 1):]
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
        day_real = result_day[:,:,:(self.hp.seasonal_window//2 + 1)]
        day_imag = result_day[:,:,(self.hp.seasonal_window//2 + 1):]
        f_day = torch.stack((day_real,day_imag),dim=-1)
        f_day = torch.view_as_complex(f_day)
        result_day = torch.fft.irfft(f_day)
        day_mu = self.day_mu(result_day)
        day_log_var = self.day_log_var(result_day)

        #特征融合
        return hour_mu, hour_log_var, day_mu, day_log_var
    
    def forward(self, input, input_normal, mode, mask):
        # mask_copy = mask.clone()
        # mask_copy[:,-1] = 1
        # mask_copy = mask_copy.unsqueeze(1)
        # input = input*mask_copy
        """
        前向传播
        :param input: 输入张量，shape = (batch, 1, window)
        """
        if mode == "train":
            loss = self.loss_func(input, input_normal, mask)
            return loss
        elif mode == "valid":
            loss = self.loss_func(input, input_normal, mask)
            return loss
        else:
            return self.reference(input, input_normal)

    def loss_func(self, input, input_normal, mask):
        trend_signal = self.series_decop(input)

        hour_mu, hour_log_var, day_mu, day_log_var = self.encode(input)
        trend_day = trend_signal[:,:,-self.hp.seasonal_window:].squeeze(1)
        trend_hour = trend_signal[:,:,-self.hp.contextual_window:].squeeze(1)
        input_day = input_normal[:,:,-self.hp.seasonal_window:].squeeze(1)
        input_hour = input_normal[:,:,-self.hp.contextual_window:].squeeze(1)
        mask_day = mask[:, -self.hp.seasonal_window:]
        mask_hour = mask[:, -self.hp.contextual_window:]
        hour_mu = hour_mu.squeeze(1)
        hour_log_var = hour_log_var.squeeze(1)
        day_mu = day_mu.squeeze(1)
        day_log_var = day_log_var.squeeze(1)

        input_day = input_day*mask_day + trend_day*torch.logical_not(mask_day)
        input_hour = input_hour*mask_hour + trend_hour*torch.logical_not(mask_hour)

        hour_recon_loss = torch.mean(
            0.5*torch.mean((hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var)),dim=-1),
            dim = 0
        )


        day_recon_loss = torch.mean(
            0.5*torch.mean((day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var)),dim=-1),
            dim = 0
        )

        recon_loss = day_recon_loss + hour_recon_loss


        loss = recon_loss
        if torch.isinf(loss):
            raise
        return loss

        
    
    def reference(self,input, input_normal):
        loss = 0
        input_hour = input_normal[:,:,-self.hp.contextual_window:]
        input_day = input_normal[:,:,-self.hp.seasonal_window:]
        hour_mu, hour_log_var, day_mu, day_log_var = self.encode(input)

        hour_loss = (hour_log_var + (input_hour - hour_mu)**2/torch.exp(hour_log_var))
        hour_loss = hour_loss[:,:,-1].unsqueeze(2)

        cycle_loss = (day_log_var + (input_day - day_mu)**2/torch.exp(day_log_var))
        cycle_loss = cycle_loss[:,:,-1].unsqueeze(2)

        loss =  hour_loss + cycle_loss

        recon_x = (hour_mu[:,:,-1].unsqueeze(2) + day_mu[:,:,-1].unsqueeze(2))/2
        return loss, recon_x

    
    def get_freq(self, input):
        f_global = torch.fft.rfft(input, dim = -1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        return f_global

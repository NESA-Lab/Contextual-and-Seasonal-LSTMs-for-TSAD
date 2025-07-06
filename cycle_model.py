# import torch
# import torch.nn as nn

# class CycleVAE(nn.Module):
#     def __init__(self, hp):
#         super(CycleVAE,self).__init__()
#         self.hp = hp
        

#         #对前序窗口特征融合得到 condition
#         self.condition_layer = nn.Sequential(
#             nn.Conv1d(in_channels=7,out_channels=1,kernel_size=1),
#             nn.Tanh()
#         )

#         self.condition_emb = nn.Sequential(
#             nn.Linear(self.hp.input_dim, self.hp.condition_dim),
#             nn.Tanh()
#         )


#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(self.hp.input_dim + self.hp.condition_dim,128),
#             nn.Tanh(),
#             nn.Linear(128,128),
#             nn.Tanh()
#         )

#         # 隐变量
#         self.mu = nn.Linear(128, self.hp.latent_dim)
#         self.log_var = nn.Linear(128, self.hp.latent_dim)

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(self.hp.latent_dim + self.hp.condition_dim,128),
#             nn.Tanh(),
#             nn.Linear(128,128),
#             nn.Tanh(),
#             nn.Linear(128, self.hp.input_dim),
#             nn.Tanh()
#         )

#         # 重构
#         self.mu_x = nn.Linear(self.hp.input_dim,self.hp.input_dim)
#         self.log_var_x = nn.Linear(self.hp.input_dim,self.hp.input_dim)
        

#     def reparaterize(self, mu, log_var):
#         std = torch.sqrt(torch.exp(log_var))
#         eps = torch.randn_like(std)
#         return eps * std + mu
    
#     def encode(self, x, c):
#         x_cond = torch.cat((x,c),dim=-1)
#         result = self.encoder(x_cond)
#         result = result.view(result.size(0),-1)
#         mu = self.mu(result)
#         log_var = self.log_var(result)
#         return mu, log_var
    
#     def decode(self, z, c):
#         c = c.view(c.size(0),-1)
#         z_cond = torch.cat((z,c),dim=-1)
#         result = self.decoder(z_cond)
#         result = result.unsqueeze(1)
#         return result
    
#     def forward(self, input):
#         #取最后一天 (batch,1,24)
#         x = input[:,-1,:].unsqueeze(1)
#         #取前面几天
#         c = input[:,:-1,:]
#         # f_c = torch.fft.rfft(c,dim=-1)
#         # c = torch.cat([c,f_c.real,f_c.imag],dim=-1)
#         #条件融合
#         # c = c.view(c.shape[0], 1, -1)
#         c = self.condition_layer(c)
#         c = self.condition_emb(c)
#         mu, log_var = self.encode(x, c)
#         z = self.reparaterize(mu, log_var)
#         result = self.decode(z, c)
#         mu_x = self.mu_x(result)
#         log_var_x = self.log_var_x(result)
#         kl_loss = self.kl_loss(mu, log_var)
#         return mu_x, log_var_x, kl_loss
    
#     def kl_loss(self, mu, log_var):
#         kl_loss = torch.mean(
#             -0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=-1),
#             dim=0
#         )
#         return kl_loss
    



# class Filter_net(nn.Module):
#     def __init__(self, num, sample_rate, hp):
#         super(Filter_net,self).__init__()
#         self.hp = hp
#         self.cutoff_frequencies = nn.Parameter(torch.linspace(1.0, 5.0, steps=num)).to(self.hp.gpu)
#         self.sample_rate = sample_rate
#         # self.fusion_layer = nn.Sequential(
#         #     nn.Conv1d(in_channels=num+1, out_channels=1, kernel_size=1),
#         #     nn.Tanh()
#         # )

#     def low_pass_filter(self, freqs, freq_domain, cutoff_frequency):
#         mask = freqs<=cutoff_frequency
#         filtered_freq_domain = freq_domain * mask
#         return torch.fft.irfft(filtered_freq_domain, n=self.hp.input_dim)
    
#     def forward(self, data):
#         freq_domain = torch.fft.rfft(data,dim=-1)
#         freqs = torch.fft.rfftfreq(data.size(-1),d=1/self.sample_rate).to(self.hp.gpu)
#         filtered_sequences = [
#             self.low_pass_filter(freqs, freq_domain, c) for c in self.cutoff_frequencies 
#         ]
#         filtered_sequences.append(data)
#         filtered_sequences = torch.cat(filtered_sequences,dim=1)
#         # result = self.fusion_layer(filtered_sequences)
#         return filtered_sequences

    
# class KalmanFilterNetwork(nn.Module):
#     def __init__(self, hp):
#         super(KalmanFilterNetwork,self).__init__()
#         self.hp = hp
#         # 可学习的卡尔曼滤波参数
#         self.Q = nn.Parameter(torch.tensor(0.01)).to(hp.gpu)  # 过程噪声
#         self.R = nn.Parameter(torch.tensor(0.1)).to(hp.gpu)   # 观测噪声
#         self.F = nn.Parameter(torch.tensor(1.0)).to(hp.gpu)   # 状态转移矩阵
#         self.H = nn.Parameter(torch.tensor(1.0)).to(hp.gpu)   # 观测矩阵
    
#     def forward(self, data):
#         data = data.squeeze(1)
#         batch, length = data.shape
#         x_hat = torch.zeros(batch, length).to(self.hp.gpu)
#         #初始误差协方差
#         P = torch.ones(batch, length).to(self.hp.gpu)

#         for t in range(1, length):
#             # 预测
#             x_pred = self.F * x_hat[:, t-1]
#             P_pred = P[:, t-1] + self.Q

#             #更新
#             K = P_pred /(P_pred + self.R) #卡尔曼增益
#             x_hat[:, t] = x_pred + K * (data[:,t] - self.H * x_pred)
#             P[:,t] = (1 - K * self.H) * P_pred
        
#         x_hat=x_hat.unsqueeze(1)

#         return x_hat







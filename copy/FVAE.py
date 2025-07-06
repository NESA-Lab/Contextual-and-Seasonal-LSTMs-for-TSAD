import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
from Layers import MLPEncoder, MLPDecoder
import pywt

class FVAE(nn.Module):
    def __init__(
            self,
            hp,
            max_capacity: int = 25,
            Capacity_max_iter: int = 1e5,
            loss_type: str = "C"
    ):
        super(FVAE,self).__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.hidden_dims = [100,100]

        #编码器
        self.encoder = MLPEncoder(input_dim=self.hp.window + 2, hidden_dims=self.hidden_dims)
        self.fc_mu = nn.Linear(self.hidden_dims[-1],self.hp.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )
        #解码器
        self.decoder = MLPDecoder(input_dim=self.hp.latent_dim,
                                  hidden_dims=self.hidden_dims,
                                  output_dim=self.hp.window + 2)
        self.fc_mu_x = nn.Linear(self.hp.window + 2, self.hp.window + 2)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window +2, self.hp.window + 2), nn.Softplus()
        )

    def encode(self, input):
        x = input
        f_x = torch.fft.rfft(x)
        f_x = torch.cat((f_x.real,f_x.imag),dim=-1)
        result = self.encoder(f_x)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return [mu, var]
    
    def decode(self, z):
        result = self.decoder(z)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,input,mode,mask):
        if mode == "train" or mode == "valid":
            mu, var = self.encode(input)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z)
            loss = self.loss_func(mu_x, var_x, input, mu, var, mask, z)
            return loss
        else:
            return self.MCMC2(input)

    def loss_func(self, mu_x, var_x, input, mu, var, mask, z):
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        mu = mu.squeeze(1)
        var = var.squeeze(1)
        z = z.squeeze(1)
        input = input.squeeze(1)

        #掩盖掉异常点，保留正常的频率信息
        input = input*mask
        recon_loss = 0
        for i in range(len(input)):
            input_i = input[i]
            input_i = input_i[torch.where(input_i!=0)]
            f_input_i = torch.fft.rfft(input_i)
            mu_x_i = mu_x[i]
            l = len(f_input_i)
            mu_x_i_real = mu_x_i[:l]
            mu_x_i_imag = mu_x_i[(len(mu_x_i)//2-1):(len(mu_x_i)//2-1+l)]
            mu_x_i = torch.cat((mu_x_i_real,mu_x_i_imag))
            f_input_i = torch.cat((f_input_i.real,f_input_i.imag))
            recon_loss += torch.mean(
                (mu_x_i-f_input_i)**2
            )

        # f_input = torch.fft.rfft(input)
        # f_input = torch.cat((f_input.real,f_input.imag),dim=-1)
        # recon_loss = torch.mean(
        #     0.5*
        #     torch.mean(torch.log(var_x) + (f_input - mu_x)**2/var_x, dim=1),
        #     dim=0,
        # )
        recon_loss /= len(input)

        # m = (torch.sum(mask, dim=1) / self.hp.window)
        kld_loss= torch.mean(
            -0.5*torch.sum(1+torch.log(var)-mu**2-var,dim=1),
            dim=0
        )
        # m = (torch.sum(mask, dim=1, keepdim=True) / self.hp.window).repeat(
        #     1, self.hp.latent_dim
        # )

        # kld_loss = torch.mean(
        #     0.5 * 
        #     torch.mean(m * (z**2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
        #     dim = 0,
        # )
        
        loss = recon_loss + kld_loss
        return loss
        
    def MCMC2(self,input):
        mu, var = self.encode(input)
        loss_all = 0
        for i in range(128):
            z=self.reparameterize(mu,var)
            mu_x,var_x=self.decode(z)
            #计算没有最后一点时的频率分布
            # x=input.clone()
            # x[:,:,-1]=0
            # f_without_last = torch.fft.rfft(x)
            # f_without_last = torch.cat((f_without_last.real,f_without_last.imag),dim=-1)
            # #计算有最后一点时的频率分布
            # x=input.clone()
            # f_with_last = torch.fft.rfft(x)
            # f_with_last = torch.cat((f_with_last.real,f_with_last.imag),dim=-1)
            # #计算损失
            # loss_without_last = torch.mean(
            #     -0.5 * (torch.log(var_x) + (f_without_last-mu_x)**2/var_x),
            #     dim = -1,
            # )
            # loss_with_last = torch.mean(
            #     -0.5 * (torch.log(var_x) + (f_with_last-mu_x)**2/var_x),
            #     dim = -1,
            # )
            # loss_all += loss_with_last - loss_without_last
            l = mu_x.shape[-1]
            f_mu_x = torch.stack((mu_x[:,:,:(l//2)],mu_x[:,:,(l//2):]),dim=-1)
            f_mu_x = torch.view_as_complex(f_mu_x)
            t_mu_x = torch.fft.irfft(f_mu_x)
            loss_all+= (t_mu_x-input)**2
        #只保留最后一个点的误差
        loss_all = loss_all[:,:,-1]
        return loss_all/128




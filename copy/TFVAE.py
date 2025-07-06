import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
import pywt
from TCN import TCN
from Layers import MLPEncoder, MLPDecoder,ConvFusion,WeightedFusion


class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super(CVAE, self).__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.hidden_dims=[100,100]

        #频域
        self.freq_encoder = MLPEncoder(input_dim = self.hp.window + 2*self.hp.condition_emb_dim, hidden_dims=[100,100])
        self.freq_fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.freq_fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )

        self.freq_decoder = MLPDecoder(input_dim = self.hp.latent_dim + 2*self.hp.condition_emb_dim,hidden_dims=[100,100],output_dim=self.hp.window)
        self.freq_fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.freq_fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), 
            nn.Softplus()
        )

        #时域
        self.time_encoder = MLPEncoder(input_dim = self.hp.window + self.hp.condition_emb_dim, hidden_dims=[100,100])
        self.time_fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.time_fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )

        self.time_decoder = MLPDecoder(input_dim = self.hp.latent_dim + self.hp.condition_emb_dim,hidden_dims=[100,100],output_dim=self.hp.window)
        self.time_fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.time_fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), 
            nn.Softplus()
        )

        # #统计
        # self.sta_encoder = MLPEncoder(input_dim = self.hp.window + 6, hidden_dims=[100,100])
        # self.sta_fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        # self.sta_fc_var = nn.Sequential(
        #     nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
        #     nn.Softplus(),
        # )

        # self.sta_decoder = MLPDecoder(input_dim = self.hp.latent_dim + 6,hidden_dims=[100,100],output_dim=self.hp.window)
        # self.sta_fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        # self.sta_fc_var_x = nn.Sequential(
        #     nn.Linear(self.hp.window, self.hp.window), 
        #     nn.Softplus()
        # )

        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )
        self.emb_local = nn.Sequential(
            nn.Linear(2 + self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.tcn=TCN(in_channels=1,out_channels=32,kernel_size=3,num_layers=4)
        self.emb_tcn=nn.Sequential(
            nn.Linear(self.hp.window-1,self.hp.condition_emb_dim),
            nn.Tanh(),
        )

        #特征融合
        self.cf=ConvFusion(input_channels=2,output_dim=self.hp.window)
        self.wf=WeightedFusion()

    def encode(self, input, condition_type):
        if condition_type == 0:
            result = self.freq_encoder(input)
            result = torch.flatten(result, start_dim=1)
            mu = self.freq_fc_mu(result)
            var = self.freq_fc_var(result)
        elif condition_type == 1:
            result = self.time_encoder(input)
            result = torch.flatten(result, start_dim=1)
            mu = self.time_fc_mu(result)
            var = self.time_fc_var(result)
        elif condition_type == 2:
            result = self.sta_encoder(input)
            result = torch.flatten(result, start_dim=1)
            mu = self.sta_fc_mu(result)
            var = self.sta_fc_var(result)
        return [mu, var]

    def decode(self, z, condition_type):
        if condition_type == 0:
            result = self.freq_decoder(z)
            mu_x = self.freq_fc_mu_x(result)
            var_x = self.freq_fc_var_x(result)
        elif condition_type == 1:
            result = self.time_decoder(z)
            mu_x = self.time_fc_mu_x(result)
            var_x = self.time_fc_var_x(result)
        elif condition_type == 2:
            result = self.sta_decoder(z)
            mu_x = self.sta_fc_mu_x(result)
            var_x = self.sta_fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode == "train" or mode == "valid":
            f_condition=self.get_freq_condition(input)
            f_condition=self.dropout(f_condition)
            f_mu,f_var=self.encode(torch.cat((input,f_condition),dim=2),condition_type=0)
            f_z = self.reparameterize(f_mu, f_var)
            f_mu_x, f_var_x = self.decode(torch.cat((f_z, f_condition.squeeze(1)), dim=1),condition_type=0)
            f_rec_x=self.reparameterize(f_mu_x,f_var_x)

            t_condition=self.get_time_condition(input)
            t_mu,t_var=self.encode(torch.cat((input,t_condition),dim=2),condition_type=1)
            t_z = self.reparameterize(t_mu, t_var)
            t_mu_x, t_var_x = self.decode(torch.cat((t_z, t_condition.squeeze(1)), dim=1),condition_type=1)
            t_rec_x=self.reparameterize(t_mu_x,t_var_x)

            # s_condition=self.get_statistic_condition(input)
            # s_mu,s_var=self.encode(torch.cat((input,s_condition),dim=2),condition_type=2)
            # s_z = self.reparameterize(s_mu, s_var)
            # s_mu_x, s_var_x = self.decode(torch.cat((s_z, s_condition.squeeze(1)), dim=1),condition_type=2)
            # s_rec_x=self.reparameterize(s_mu_x,s_var_x)

            rec_x=self.cf(f_rec_x,t_rec_x)
            # rec_x=self.wf(f_rec_x,t_rec_x)

            f_recon_loss=self.reconstruction_loss(f_mu_x,f_var_x,input,y)
            f_kl_loss=self.kld_loss(f_mu,f_var,y,f_z)
            t_recon_loss=self.reconstruction_loss(t_mu_x,t_var_x,input,y)
            t_kl_loss=self.kld_loss(t_mu,t_var,y,t_z)
            # s_recon_loss=self.reconstruction_loss(s_mu_x,s_var_x,input,y)
            # s_kl_loss=self.kld_loss(s_mu,s_var,y,s_z)
            f_loss=self.fussion_loss(rec_x,input,y)
            loss=f_loss+f_recon_loss+t_recon_loss+f_kl_loss+t_kl_loss
            return rec_x,loss
        else:
            y = y.unsqueeze(1)
            return self.MCMC2(input)

    def get_freq_condition(self, x):
        x_g = x
        f_global = torch.fft.rfft(x_g[:, :, :-1], dim=-1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        f_global = self.emb_global(f_global)
        x_g = x_g.view(x.shape[0], 1, 1, -1)
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.transpose(1, 2)
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        f_local = f_local[:, -1, :].unsqueeze(1)
        output = torch.cat((f_global, f_local), -1)
        # output=f_local
        return output
    
    def get_time_condition(self,x):
        x_t=x
        x_t=self.tcn(x_t[:,:,:-1])
        x_t=self.emb_tcn(x_t)
        # x_s=self.get_statistic_condition(x)
        output = x_t
        return output
    
    def get_statistic_condition(self,x):
        max_value=torch.max(x,dim=-1).values
        min_value=torch.min(x,dim=-1).values
        tp90=torch.quantile(x,q=0.9,dim=-1)
        tm90=torch.quantile(x,q=0.1,dim=-1)
        mean_value=torch.mean(x,dim=-1)
        var_value=torch.var(x,dim=-1)
        output=torch.cat([max_value,min_value,tp90,tm90,mean_value,var_value],dim=-1)
        output=output.view(x.shape[0],1,-1)
        return output
    

    def MCMC2(self, x):
        f_condition=self.get_freq_condition(x)
        t_condition=self.get_time_condition(x)
        # s_condition=self.get_statistic_condition(x)
        origin_x = x.clone()
        #频域重构
        f_mu, f_var = self.encode(torch.cat((x, f_condition), dim=2),0)
        #时域重构
        t_mu, t_var = self.encode(torch.cat((x, t_condition), dim=2),1)
        # #统计重构
        # s_mu, s_var = self.encode(torch.cat((x, s_condition), dim=2),2)
        l = torch.ones_like(origin_x)
        l[:, :, -1] = 0
        prob_all = 0
        for i in range(128):
            #频域重构
            f_z = self.reparameterize(f_mu, f_var)
            f_mu_x, f_var_x = self.decode(torch.cat((f_z, f_condition.squeeze(1)), dim=1),0)
            f_recon_x=self.reparameterize(f_mu_x,f_var_x)
            prob_all += -0.5 * (torch.log(f_var_x) + (origin_x - f_mu_x) ** 2 / f_var_x)
            #时域重构
            t_z = self.reparameterize(t_mu, t_var)
            t_mu_x, t_var_x = self.decode(torch.cat((t_z, t_condition.squeeze(1)), dim=1),1)
            t_recon_x=self.reparameterize(t_mu_x,t_var_x)
            prob_all += -0.5 * (torch.log(t_var_x) + (origin_x - t_mu_x) ** 2 / t_var_x)
            # # 统计重构
            # s_z = self.reparameterize(s_mu, s_var)
            # s_mu_x, s_var_x = self.decode(torch.cat((s_z, s_condition.squeeze(1)), dim=1),2)
            # s_recon_x=self.reparameterize(s_mu_x,s_var_x)
            # prob_all += -0.5 * (torch.log(s_var_x) + (origin_x - s_mu_x) ** 2 / s_var_x)
            #特征融合
            recon_x=self.cf(f_recon_x,t_recon_x)
            # recon_x=self.wf(f_recon_x,t_recon_x)
            recon_x=l*recon_x+origin_x
            prob_all+=(recon_x-origin_x)**2

        return recon_x, prob_all / 128
    
    def reconstruction_loss(self, mu_x, var_x, input, y, mode="nottrain"):
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        input = input.squeeze(1)

        recon_loss = y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x)
        nonzero_sum = torch.sum(recon_loss, dim=1)
        nonzero_count = torch.sum(recon_loss!=0, dim=1)
        nonzero_count = torch.where(nonzero_count == 0, torch.tensor(1).to("cuda"), nonzero_count)
        recon_loss=torch.mean(nonzero_sum/nonzero_count,dim=0)
        return recon_loss

    def kld_loss(self, mu, var, y, z, mode="nottrain"):
        m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
            1, self.hp.latent_dim
        )
        #KL误差
        kld_loss = torch.mean(
            0.5 * torch.mean(m * (z**2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
            dim=0,
        )
        # m = (torch.sum(y, dim=1) / self.hp.window)

        # kld_loss=torch.mean(
        #     -0.5*torch.sum(1+torch.log(var)-mu**2-var,dim=1)*m
        # )

        return kld_loss

    def fussion_loss(self,recon_x,input,y):
        recon_x=recon_x.squeeze(1)
        input=input.squeeze(1)
        f_loss = y * ((recon_x - input)**2)
        nonzero_sum = torch.sum(f_loss, dim=1)
        nonzero_count = torch.sum(f_loss!=0, dim=1)
        nonzero_count = torch.where(nonzero_count == 0, torch.tensor(1).to("cuda"), nonzero_count)
        f_loss = torch.mean(nonzero_sum/nonzero_count,dim=0)
        return f_loss
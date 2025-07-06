import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
import pywt
from TCN import TCN
from Layers import MLPEncoder, MLPDecoder,ConvFusion, WeightedFusion


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
        # self.freq_encoder = MLPEncoder(input_dim = self.hp.window, hidden_dims=[100,100])
        # self.freq_encoder = MLPEncoder(input_dim = self.hp.condition_emb_dim, hidden_dims=[100,100])
        # self.freq_fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        # self.freq_fc_var = nn.Sequential(
        #     nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
        #     nn.Softplus(),
        # )

        # self.freq_decoder = MLPDecoder(input_dim = self.hp.latent_dim, hidden_dims=[100,100],output_dim=self.hp.window+2)
        # self.freq_fc_mu_x = nn.Linear(self.hp.window+2, self.hp.window+2)
        # self.freq_fc_var_x = nn.Sequential(
        #     nn.Linear(self.hp.window+2, self.hp.window+2), 
        #     nn.Softplus()
        # )
        
        # self.freq_decoder = MLPDecoder(input_dim = self.hp.latent_dim, hidden_dims=[100,100],output_dim=self.hp.condition_emb_dim)
        # self.freq_fc_mu_x = nn.Linear(self.hp.condition_emb_dim, self.hp.condition_emb_dim)
        # self.freq_fc_var_x = nn.Sequential(
        #     nn.Linear(self.hp.condition_emb_dim, self.hp.condition_emb_dim), 
        #     nn.Softplus()
        # )

        #时域
        self.time_encoder = MLPEncoder(input_dim = self.hp.window, hidden_dims=[100,100])
        self.time_fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.time_fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )

        self.time_decoder = MLPDecoder(input_dim = self.hp.latent_dim, hidden_dims=[100,100],output_dim=self.hp.window)
        self.time_fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.time_fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), 
            nn.Softplus()
        )

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
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.condition_emb_dim),
            nn.Tanh(),
        )

        self.emb_local = nn.Sequential(
            nn.Linear(2 + self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.tcn=TCN(in_channels=1,out_channels=32,kernel_size=3,num_layers=4)
        self.emb_tcn=nn.Sequential(
            nn.Linear(self.hp.window-1,self.hp.condition_emb_dim),
            nn.Tanh(),
        )

        #特征融合
        self.cf=ConvFusion(input_channels=2,output_dim=self.hp.window)

    def encode(self, input, type):
        if type == 0:
            # x=input[:,:,:-1]
            # x=torch.fft.rfft(x,dim=-1)
            # x=torch.cat((x.real,x.imag),dim=-1)
            x=self.get_freq_condition(input)
            result = self.freq_encoder(x)      
            result= torch.flatten(result,start_dim=1)
            mu=self.freq_fc_mu(result)
            var=self.freq_fc_var(result)
        elif type == 1:
            x=input
            # x_tcn=self.get_time_condition(input)
            # x=torch.cat((x,x_tcn),dim=-1)
            result = self.time_encoder(x)
            result = torch.flatten(result, start_dim=1)
            mu = self.time_fc_mu(result)
            var = self.time_fc_var(result)
        return [mu, var]

    def decode(self, z, type):
        if type == 0:
            result = self.freq_decoder(z)
            mu_x = self.freq_fc_mu_x(result)
            var_x = self.freq_fc_var_x(result)
        elif type == 1:
            result = self.time_decoder(z)
            mu_x = self.time_fc_mu_x(result)
            var_x = self.time_fc_var_x(result)
        return mu_x, var_x
    
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
    
    # def get_time_condition(self,x):
    #     x_t=x
    #     x_t=self.tcn(x_t[:,:,:-1])
    #     x_t=self.emb_tcn(x_t)
    #     output = x_t
    #     return output

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode == "train" or mode == "valid":

            t_mu,t_var=self.encode(input,type=1)
            t_z = self.reparameterize(t_mu, t_var)
            t_mu_x, t_var_x = self.decode(t_z, type=1)
            t_rec_x=self.reparameterize(t_mu_x,t_var_x)

            t_recon_loss=self.reconstruction_loss(t_mu_x,input,y)
            t_freq_loss=self.freq_loss(t_mu_x,input,y)
            t_kl_loss=self.kld_loss(t_mu,t_var,y,t_z)

            loss=t_recon_loss+t_kl_loss+t_freq_loss
            return t_rec_x,loss
        else:
            y = y.unsqueeze(1)
            return self.MCMC2(input)


    def MCMC2(self, x):
        origin_x = x.clone()
        # #频域重构
        # f_mu, f_var = self.encode(x,0)
        #时域重构
        t_mu, t_var = self.encode(x,1)
        l = torch.ones_like(origin_x)
        l[:, :, -1] = 0
        prob_all = 0
        for i in range(128):

            #时域重构
            t_z = self.reparameterize(t_mu, t_var)
            t_mu_x, t_var_x = self.decode(t_z,1)
            # t_recon_x=self.reparameterize(t_mu_x,t_var_x)
            prob_all += -0.5 * (torch.log(t_var_x) + (origin_x - t_mu_x) ** 2 / t_var_x)
            # prob_all+=((t_mu_x-origin_x)**2)

            t_mu_x=(1-l)*t_mu_x+l*origin_x
            f_mu_x=torch.fft.rfft(t_mu_x)
            f_mu_x=torch.cat((f_mu_x.real,f_mu_x.imag),dim=-1)
            f_input=torch.fft.rfft(origin_x)
            f_input=torch.cat((f_input.real,f_input.imag),dim=-1)
            prob_all+= torch.mean((f_input-f_mu_x)**2,keepdim=True,dim=-1).repeat(1,1,self.hp.window)

            #fussion
            # recon_x=self.wf(f_recon_x,t_recon_x)


        return t_mu_x, prob_all / 128
    
    def reconstruction_loss(self, recon_x, input, y, mode="nottrain"):
        # mu_x = mu_x.squeeze(1)
        # var_x = var_x.squeeze(1)
        recon_x=recon_x.squeeze(1)
        input = input.squeeze(1)

        # recon_loss = 0.5* y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x)
        recon_loss =  y * ((recon_x-input)**2)
        nonzero_sum = torch.sum(recon_loss, dim=1)
        nonzero_count = torch.sum(recon_loss!=0, dim=1)
        nonzero_count = torch.where(nonzero_count == 0, torch.tensor(1).to("cuda"), nonzero_count)
        recon_loss=torch.mean(nonzero_sum/nonzero_count,dim=0)
        return recon_loss

    def kld_loss(self, mu, var, y, z):
        # m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
        #     1, self.hp.latent_dim
        # )
        # # KL散度
        # kld_loss = torch.mean(
        #     0.5 * torch.mean(m * (z**2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
        #     dim=0,
        # )
        m = (torch.sum(y, dim=1) / self.hp.window)

        kld_loss=torch.mean(
            -0.5*torch.mean(1+torch.log(var)-mu**2-var,dim=1)*m,
            dim=0
        )

        return kld_loss
    
    def freq_loss(self,recon_x,input,y):
        recon_x=recon_x.squeeze(1)
        input = input.squeeze(1)

        l=torch.zeros_like(input)
        l[:,-1]=0
        #只保留最后一点
        recon_x=(1-l)*recon_x+l*input
        
        f_recon_x=torch.fft.rfft(recon_x)
        f_recon_x=torch.cat((f_recon_x.real,f_recon_x.imag),dim=-1)
        f_input=torch.fft.rfft(input)
        f_input=torch.cat((f_input.real,f_input.imag),dim=-1)

        freq_loss=torch.mean(
            torch.mean((f_recon_x-f_input)**2,dim=1),
            dim=0
        )
        return freq_loss
    
    
    def get_freq_mu_var(self, mu_x, var_x):
        l=self.hp.window//2+1

        mu_real=mu_x[:,:,:l]
        mu_imag=mu_x[:,:,l:]
        mu_x=torch.stack((mu_real,mu_imag),dim=-1)
        mu_x=torch.view_as_complex(mu_x)
        mu_x=torch.fft.irfft(mu_x,dim=-1).real

        var_real=var_x[:,:,:l]
        var_imag=var_x[:,:,l:]
        var_x=torch.stack((var_real,var_imag),dim=-1)
        var_x=torch.view_as_complex(var_x)
        var_x=torch.fft.irfft(var_x,dim=-1).real
        var_x=nn.functional.softplus(var_x)

        return mu_x,var_x
# import torch
# import torch.nn as nn

# class AnomalyLSTM(nn.Module):
#     def __init__(self, hp):
#         super(AnomalyLSTM, self).__init__()
#         self.hp = hp
#         self.encoder = nn.LSTM(1, self.hp.hidden_dim, batch_first = True)
#         self.decoder = nn.LSTM(1, self.hp.hidden_dim, batch_first = True)
#         self.fc = nn.Sequential(
#             nn.Linear(self.hp.hidden_dim, 1),
#             nn.Tanh()
#         )
#         self.mu = nn.Linear(self.hp.future, self.hp.future)
#         self.logvar = nn.Linear(self.hp.future, self.hp.future)


#     def get_input(self, input):
#         batch, window = input.shape
#         seq_input = input[:,:-self.hp.future]
#         gt = input[:,-self.hp.future:]
#         return seq_input, gt

#     def forward(self, input, mode, mask):
#         seq_input, gt = self.get_input(input)
#         #添加 dim 维度
#         batch, length = seq_input.shape
#         seq_input = seq_input.unsqueeze(2)
#         mask = mask[:,-self.hp.future:]

#         _, (h, c) = self.encoder(seq_input)
#         decoder_input = torch.zeros((batch, 1, 1)).to(self.hp.gpu)
#         outputs = []
#         for _ in range(self.hp.future):
#             out, (h, c) = self.decoder(decoder_input, (h,c))
#             out = self.fc(out)
#             outputs.append(out)
#             decoder_input = out
        
#         result = torch.cat(outputs, dim=1).squeeze(-1)
#         mu = self.mu(result)
#         logvar = self.logvar(result)

#         if mode == "train" or mode == "valid":
#             return self.loss_function(mu, logvar, gt, mask)
#         else:
#             return self.predict(mu, logvar, gt)


#     def loss_function(self, mu, logvar, gt, mask):
#         loss = torch.mean(
#             torch.mean((logvar + (mu - gt)**2/torch.exp(logvar)),dim=-1),
#             dim = 0
#         )
#         return loss
    
    
#     def predict(self, mu, logvar, gt):
#         return logvar + (mu - gt)**2/torch.exp(logvar)

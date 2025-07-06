import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt

class QuantileModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 3))  # 0.05, 0.5, 0.95

    def forward(self, x):
        return self.fc(x)
    
    def quantile_loss(preds, target, quantiles=[0.05,0.5,0.95]):
        loss = 0
        for i, q in enumerate(quantiles):
            errors = target - preds[:,i].unsqueeze(1)
            loss += torch.mean(torch.max(q*errors, (q-1)*errors))
        return loss
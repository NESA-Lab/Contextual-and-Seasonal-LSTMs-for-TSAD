import pywt
import torch.nn as nn
import torch
import numpy as np

class Series_decop(nn.Module):
    def __init__(self, wavlet='db4', level=5, hp=None):
        super().__init__()
        self.wavlet = wavlet
        self.level = level
        self.hp = hp

    def forward(self, x):
        signal = x.clone().cpu().detach().numpy()
        coeffs = pywt.wavedec(signal[:,:], self.wavlet, level=self.level)
        # 估计噪声阈值
        eps = 1e-10
        d = np.sqrt(2*np.log(signal.shape[-1]))
        for i in range(1, self.level+1):
            sigma = np.median(np.abs(coeffs[i])) / 0.6745
            sigma = max(sigma, eps)
            uthresh = sigma*d
            # 对细节系数应用软阈值
            coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode='soft')
    
        if self.hp.use_gpu == True:
            trend_signal = torch.tensor(pywt.waverec(coeffs, self.wavlet)).to(self.hp.gpu)
        else:
            trend_signal = torch.tensor(pywt.waverec(coeffs, self.wavlet))

        return trend_signal

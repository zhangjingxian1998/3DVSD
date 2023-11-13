import torch
import torch.nn as nn
from vsd_3d.models import FFN

class EDGESCORE(nn.Module):
    def __init__(self):
        super(EDGESCORE,self).__init__()
        self.calculate_score = FFN(in_feature=768,middle_feature=3072,out_feature=768)
        self.score = nn.Linear(768,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.calculate_score(x)
        x = self.score(x)
        x = self.sigmoid(x)
        return x
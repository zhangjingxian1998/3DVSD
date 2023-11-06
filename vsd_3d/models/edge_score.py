import torch
import torch.nn as nn
from vsd_3d.models import FFN

class EDGESCORE(nn.Module):
    def __init__(self,middle_feature=768):
        super(EDGESCORE,self).__init__()
        self.calculate_score = FFN(middle_feature,out_feature=1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        x = self.calculate_score(x)
        x = self.sigmoid(x)
        return x
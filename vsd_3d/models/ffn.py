import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self,in_feature=768, middle_feature=768, out_feature=768, d_prob=0.1):
        super(FFN,self).__init__()
        self.linear1 = nn.Linear(in_feature,middle_feature)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(middle_feature,out_feature)
        # self.dropout = nn.Dropout(d_prob)
    def forward(self,x):
        x = self.linear1(x)
        x = self.gelu(x)
        # x = self.dropout(x) # 也会导致训练Loss不下降
        x = self.linear2(x)
        return x
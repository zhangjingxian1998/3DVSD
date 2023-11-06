import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import time
class MeanPool(nn.Module):
    def __init__(self):
        '''
        计算方法在公式3与附录A的公式4, 不清楚r_G的具体形状
        '''
        super(MeanPool,self).__init__()

    def forward(self,s_v, s_e, sub_graph,s_e_score,num_node):
        B, N, D = s_v.shape
        score = s_e_score[:,:2]                                             # 只需要考虑sub和obj
        lam_b = torch.sum(score.view(B,-1), dim=-1,keepdim=True) + 1e-6
        lam_b = lam_b.repeat(1,N)
        # 为去除没有子节点情况的干扰, 即把第一列清0
        zero_one_hot = torch.ones_like(lam_b)
        zero_one_hot[:,0] = 0

        sub_graph_mask = one_hot(sub_graph,num_classes=N)
        sub_graph_mask = sub_graph_mask * zero_one_hot      # 分子上的 +1
        
        score = torch.sum(score,dim=1) + sub_graph_mask
        lam = score / lam_b
        
        r_G = self.meanpool(s_v,s_e,lam,num_node)
        
        return r_G
    
    def meanpool(self,s_v,s_e,lam,num_node):
        B, N, D = s_v.shape
        s_v = s_v.unsqueeze(-2).repeat(1,1,N,1)                         # [B, N, D] --> [B, N, 1, D] --> [B, N, N, D]
        lam = lam.view(B,N,1,1).repeat(1,1,N,D)                         # [B,N] --> [B,N,1,1] --> [B,N,N,D]
        
        r_G = ((s_v + s_e) * lam)
        r_G = torch.sum(r_G.view(B,-1,D),dim=-2)
        r_G = r_G / torch.pow(num_node.view(B,1).repeat(1,D),2)
        return r_G.view(B,1,-1)
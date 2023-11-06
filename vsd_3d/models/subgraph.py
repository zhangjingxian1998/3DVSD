import torch
import numpy
from vsd_3d.models import EDGESCORE
import torch.nn as nn
class SUBGRAPH(nn.Module):
    def __init__(self):
        '''
        论文figure4提供的
        '''
        super(SUBGRAPH,self).__init__()
        self.calculate_score = EDGESCORE(768)
        pass
    
    def forward(self, s_e, adjacency_matrix):
        B, N, _, D = s_e.shape

        adjacency_matrix[:,0,1] = 0
        adjacency_matrix[:,1,0] = 0 # gt连边不算
        adjacency_matrix_mask_float = adjacency_matrix
        # step1
        s_e = s_e.view(B, N*N, D)                                           # 每个batch展平，送入网络处理 [B, N, N, D] --> [B, N*N, D]
        s_e_score = self.calculate_score(s_e)                               # [B, N*N, D] --> [B, N*N, 1]
        s_e_score = s_e_score.view(B, N, N) * adjacency_matrix_mask_float   # 网络处理完后，形状恢复 [B, N*N, 1] --> [B, N, N] 并且只保留存在边关系的得分

        # step2
        sub_max_score, sub_max_score_id = torch.max(s_e_score[:,0],dim=-1)  # 取sub和各边的分数，取最大值和其索引
        obj_max_score, obj_max_score_id = torch.max(s_e_score[:,1],dim=-1)  # 取obj和各边的分数，取最大值和其索引
        # step3
        sub_score_mask = (sub_max_score > obj_max_score).byte()  # 和sub连接的中间类
        obj_score_mask = (obj_max_score > sub_max_score).byte()  # 和obj连接的中间类 二者是互补的
        final_id = sub_max_score_id * sub_score_mask \
                    + obj_score_mask * obj_max_score_id
        return final_id, s_e_score
    
    def forward_(self, s_e, adjacency_matrix): # 节省显存
        B, N, _, D = s_e.shape
        s_e_score = torch.zeros_like(adjacency_matrix)
        # s_e_mask = torch.triu(graph) == 1
        # s_e_tmp = s_e[s_e_mask]
        adjacency_matrix_mask = adjacency_matrix == 1
        adjacency_matrix_mask[:,0,1] = False
        adjacency_matrix_mask[:,1,0] = False # gt连边不算
        adjacency_matrix_mask_reverse = adjacency_matrix == 0
        # step1
        s_e_tmp = s_e[adjacency_matrix_mask]
        score = self.calculate_score(s_e_tmp)
        s_e_score[adjacency_matrix_mask] = score.squeeze(-1)
        # step2
        sub_max_score, sub_max_score_id = torch.max(s_e_score[:,0],dim=-1)
        obj_max_score, obj_max_score_id = torch.max(s_e_score[:,1],dim=-1)
        # step3
        final_id = torch.zeros_like(sub_max_score_id)
        sub_score_mask = sub_max_score > obj_max_score
        obj_score_mask = obj_max_score > sub_max_score
        final_id[sub_score_mask] = sub_max_score_id[sub_score_mask]
        final_id[obj_score_mask] = obj_max_score_id[obj_score_mask]
        return final_id, s_e_score
import torch
import numpy
from vsd_3d.models import EDGESCORE
import torch.nn as nn
import copy
class SUBGRAPH(nn.Module):
    def __init__(self):
        '''
        论文figure4提供的
        '''
        super(SUBGRAPH,self).__init__()
        self.calculate_score = EDGESCORE()
        # self.softmax = nn.Softmax(dim=2)
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
        s_e_score = s_e_score[:,:2]
        # origin_score = s_e_score.view(B, N, N) - (1 - adjacency_matrix_mask_float) * 1e6
        # s_e_score = self.softmax(origin_score[:,:2,])
        # step2
        flag = torch.ones(B) * -1
        sub_max_score, sub_max_score_id = torch.max(s_e_score[:,0],dim=-1)  # 取sub和各边的分数，取最大值和其索引 [B]
        obj_max_score, obj_max_score_id = torch.max(s_e_score[:,1],dim=-1)  # 取obj和各边的分数，取最大值和其索引 [B]
        # step3 TODO 需要一个额外的状态标志     0:final_id == 0             这个标志形状应是[B,2]的, 一维上的0行表示与sub的关系, -1为没关系, 其余值为对应的与之相连的索引
                                                                                                    # 1行表示与obj的关系, -1为没关系, 其余值为对应的与之相连的索引
                                            # 1:final_id != 0 与sub相连
                                            # 2:final_id != 0 与obj相连
                                            # 3:额外判断, 有两个
        # sub_score_mask = (sub_max_score > obj_max_score).byte()  # 和sub连接的中间类
        # obj_score_mask = (obj_max_score > sub_max_score).byte()  # 和obj连接的中间类 二者是互补的
        # final_id = sub_max_score_id * sub_score_mask \
        #             + obj_score_mask * obj_max_score_id           # 这种是不对的
        mask_1 = sub_max_score_id == obj_max_score_id # 相等的地方需要对分数进行判断, 将分数高的地方保留, 处理环形情况
        sub_max_score_id_tmp = copy.deepcopy(sub_max_score_id)
        obj_max_score_id_tmp = copy.deepcopy(obj_max_score_id)
        # sub_max_score_id[mask_1] = 0
        # obj_max_score_id[mask_1] = 0
        mask_sub = (sub_max_score * mask_1) > (obj_max_score * mask_1)
        mask_obj = (obj_max_score * mask_1) > (sub_max_score * mask_1)
        sub_max_score_id[mask_sub] = sub_max_score_id_tmp[mask_sub]
        obj_max_score_id[mask_obj] = obj_max_score_id_tmp[mask_obj]
        final_id = torch.cat([sub_max_score_id.unsqueeze(-1), obj_max_score_id.unsqueeze(-1)], dim=-1) # [B, 2]
        for i, id in enumerate(final_id):
            if id[0]>0 and id[1]>0:
                flag[i] = 3
            elif id[0]>0:
                flag[i]=1
            elif id[1]>0:
                flag[i]=2
            else:
                flag[i]=0

        return final_id, s_e_score, flag
    
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
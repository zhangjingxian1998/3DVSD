import torch
import torch.nn as nn

class Loss_score(nn.Module):
    def __init__(self):
        super(Loss_score, self).__init__()

    def forward(self,output, adjacency_matrix, class_name, target_sent):
        '''
        output 是 网络计算边得分的 前两行的得分, 即与sub和obj对应边的得分
        adjacency_matrix 是邻接矩阵, 取其前两行代表sub和obj与其他目标的连接关系
        class_name 是各个目标的类别
        target_sent 是标注的描述性语句
        函数的目的是找出描述语句中存在的第三方, 使其作为sub和obj之间的关联词
        '''
        B, _, N = output.shape
        device = adjacency_matrix.device
        loss = 0
        class_name = class_name[:, 2:]                  # [B, N] --> [B, N-2]

        # 1、处理score
        score = output[:,:,2:]                         # [B, N, N] --> [B, 2, N-2]
        score_mask_float = (score == 0).float()         # 防止不存边关系部分的0做log出现-inf
                     
        score = -torch.log(score + score_mask_float)     # 将其置1, 使log值为0
        score = torch.sum(score, dim=1)                # [B, 2, N-2] --> [B, N-2]
        name_matrix = torch.zeros(B,N-2).to(device)     # [B, N-2]
        # 2、通过查询class_name每一个是否在对应的target_sent中, 给该位置赋值0|1, (0:类不在描述语句中, 1:类在描述语句中)
        for batch_id, class_name_one_batch in enumerate(class_name):
            for name_id, clas_name_one in enumerate(class_name_one_batch):
                if clas_name_one in target_sent[batch_id]:
                    name_matrix[batch_id, name_id] = 1
                    pass
        loss = torch.sum(score * name_matrix) / B
        return loss
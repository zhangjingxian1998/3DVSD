import torch
from vsd_3d.models import OcGCN, SUBGRAPH, MeanPool
import torch.nn as nn
from vsd_3d.utility.direction_rule import direction_dict
import time
import numpy as np
from vsd_3d.utility.loss import Loss_score
class Model(nn.Module):
    '''
    VSD3D处理模型
    输入3D信息,处理成图
    输出 r_G 用于视觉语言模型交叉注意力
    输出 子图方位
    '''
    def __init__(self):
        super(Model,self).__init__()
        self.OcGCN = OcGCN() # 计算图输出
        self.subgraph_creator = SUBGRAPH() # 计算子图
        self.meanpool = MeanPool()
        self.loss_score = Loss_score()
        pass

    def forward(self, args, data):
        device = next(self.parameters()).device
        self.args = args
        if args.VL_pretrain:
            data = data['batch_entry_3d']
            data['centroid'] = data['centroid'].to(device)
            class_name = data['class_name']
            # 只需要判断gt标注中的sub和obj的3D关系
            direction_list = self.calculate_direction(data['centroid'][:,:2])
            return np.array(direction_list).astype(np.object_)
        else:
            vis_feats = data['vis_feats'].to(device)
            sentence = data['batch_entry']['sentences']
            data = data['batch_entry_3d']
            data['centroid'] = data['centroid'].to(device)
            data['obj_conf'] = data['obj_conf'].to(device)
            data['basis'] = data['basis'].to(device)
            data['coeffs'] = data['coeffs'].to(device)
            class_name = data['class_name']
            # 需要判断sub和middle, middle和obj的关系
            # step1 计算邻接矩阵
            time_0 = time.time()
            adjacency_matrix = self.calculate_adjacenct_matrix(data)
            num_node = torch.sum((adjacency_matrix), dim=-1) # 计算节点数量
            num_node = torch.sum((num_node > 0).float(),dim=-1) # 计算节点数量
            time_1 = time.time()
            # step2 计算图输出
            s_v, s_e = self.OcGCN(data,vis_feats, adjacency_matrix)
            time_2 = time.time()
            # step3 子图选择，边得分 # TODO 边得分需要计算损失 如果根据要求没有找到合适的子图，没有子图连边，应该是只输入sub和obj吗，还是强制至少有一个子图
            subgraph, s_e_score = self.subgraph_creator(s_e, adjacency_matrix)
            time_3 = time.time()
            # step4 得出r_G
            r_G = self.meanpool(s_v, s_e, subgraph, s_e_score, num_node)
            time_4 = time.time()
            # TODO step5 得出方位, 用作视觉语言模型的提示词, 还需要判断是否使用模型提供的3D目标框
            B = subgraph.shape[0]
            arange = torch.arange(B)
            middle_center = data['centroid'][arange.to(device),subgraph].view(B,1,-1)
            calculate_center = torch.cat([data['centroid'][ : , 0:2], middle_center],dim=1)

            # TODO 如果有没有子图连边的情况, 需要进行判别, 生成一个mask, 关系导出时为void
            direction_list = self.calculate_direction(calculate_center)
            time_5 = time.time()
            # 返回子图中间节点类别
            subgraph_class = class_name[arange, subgraph.cpu()]

            # 计算损失值
            loss = self.loss_score(s_e_score, adjacency_matrix, class_name, sentence)
            time_6 = time.time()
            # print('计算邻接矩阵耗时:',time_1 - time_0)
            # print('计算图输出耗时:',time_2 - time_1)
            # print('子图选择耗时:',time_3 - time_2)
            # print('得出r_G耗时:',time_4 - time_3)
            # print('得出方位耗时:',time_5 - time_4)
            # print('计算损失值耗时:',time_6 - time_5)
            pass
        return r_G, np.array(direction_list).astype(np.object_), subgraph_class, loss

    def calculate_direction(self,centroid):
        '''
        输入是各个batch里的各个目标的中心坐标
        输出是各个batch种S_M和M_O的类别,形状为[B,2]
        '''
        # 如果是根据方向进行预训练，那么提供的就是两个目标
        # 如果是整体训练过程，那么就是提供2-3个目标
        # 要将位置坐标归一化，转换为[0,1], 要保持相对位置不变, 方法是坐标加上所有值的最小值，再除以各自的二范数
        B, N, D = centroid.shape
        # 坐标归一化
        min_bath = torch.min(centroid.view(B,-1),dim=-1,keepdim=True)[0]    # 找到最小值
        min_bath = torch.min(min_bath, torch.tensor(0.)) # 最小值不小于0
        min_bath = min_bath.view(B, 1, 1).repeat(1, N, D)
        center = centroid + torch.abs(min_bath) # 使最小值为0
        norm_center = torch.norm(center,dim=-1,keepdim=True)
        norm_center = norm_center.repeat(1, 1, D)
        center = center / norm_center # 完成归一化, 归一化后感觉距离相差不大了, 不知道会有什么影响 # [B,3,3]

        #取两种张量，S_M 和 M_O 方便后续运算
        if self.args.VL_pretrain:
            S_M = center[:, 0 :1]   # [B, 1, 3]
            M_O = center[:, -1: ]   # [B, 1, 3]
        else:
            S_M = torch.cat([center[:, 0:1], center[:,-1:  ]], dim=-2)  # [B, 2, 3]
            M_O = torch.cat([center[:,-1: ], center[:,-2:-1]], dim=-2)  # [B, 2, 3]
        K = S_M.shape[1]
        delta = torch.abs(S_M - M_O) # [B,2,3] 第一相减代表sub和middle 第二相减代表middle和obj
        # 如果距离差值均小于0.2，关系为next to
        delta_base = delta <= 0.2
        next_to_mask = (torch.sum(delta_base, dim=-1) != 3).byte() # 如果 <= 0.2 的个数是3个，值为0，对应next to, 否则为其他  
        # 有大量的结果都导向next to 这或许是因为都使用了模型提供的3D信息 TODO 将类别不对应的部分改为2D信息转3D
        #
        direction_index = torch.argmax(delta, dim=-1) # 如果 <= 0.2的个数不为3, 开始考虑哪个哪个方向上差值是最大的
        direction_large_mask = direction_index # 这一步用于索引用于判断属于哪个大方向 (x,y,z)
        mask = torch.zeros_like(delta, dtype=bool)
        mask = mask.scatter_(-1, direction_index.unsqueeze(-1), 1)# 为了标志出差值最大值的轴
        mask_reverse = ~mask # 取反, 标志出另外两轴
        direction_large_one_mask = (S_M[mask].view(B,-1) <= M_O[mask].view(B,-1)).byte() # 判断差值最大方向上sub和obj的空间正负关系; 
                                                                                        # sub位于obj的正方向是0, 负方向是1; 
                                                                                        # 用于区分属于x,y或z的正方向还是负方向
        
        # 判断除最大值外的两处差值大小, 如果均 <=0,2, 为基础方向,标志为0; 不均 <=0.2 的为多重方向，标志为1
        two_direction = delta[mask_reverse].view(B,-1,2) # [B, 2, 3] --> [B, 2, 2]
        base_direction_mask = (torch.sum(two_direction <= 0.2, dim=-1) != 2).byte()

        # 接下来进行重类的判断
        repeat_3_class_mask = (torch.sum(two_direction > 0.2, dim=-1) !=2).byte() # 用于区分3重类和2重类 0为3重类，1为2重类

        # 接下来处理3重类的情况
        three_repeat_mask = S_M[mask_reverse].view(B,-1,2) <= M_O[mask_reverse].view(B,-1,2) # [B,2,2]
        three_repeat_1_mask = three_repeat_mask[:,:,0].byte() # 额外的第一个轴, 为0, 代表该轴的正方向, 为1, 代表该轴的负方向
        three_repeat_2_mask = three_repeat_mask[:,:,1].byte() # 额外的第二个轴, 为0, 代表该轴的正方向, 为1, 代表该轴的负方向
        # 接下来处理2重类的情况
        repeat_2_class_mask = torch.argmax(two_direction, dim=-1) # 代表在哪个轴上存在额外关系
        mask = torch.zeros_like(delta, dtype=bool)
        mask = mask.scatter_(-1, repeat_2_class_mask.unsqueeze(-1), 1)
        two_repate_mask = (S_M[mask].view(B,-1) <= M_O[mask].view(B,-1)).byte() # 0代表该轴正方向, 1代表该轴负方向

        direction_list =[] # 作为输出
        for i in range(B):
            batch_list = []
            for j in range(K):
                result = direction_dict[str(next_to_mask[i,j].item())]      # 判断是 next to 还是其他方向
                if type(result) is str:
                    batch_list.append(result)
                    continue
                result = result[str(direction_large_mask[i,j].item())]      # 判断方向属于哪个轴向
                result = result[str(direction_large_one_mask[i,j].item())]  # 判断方向在该轴向的正负方向
                result = result[str(base_direction_mask[i,j].item())]       # 判断是否属于单一类别，如 front back up down right left
                if type(result) is str:
                    batch_list.append(result)

                    continue
                tmp = str(repeat_3_class_mask[i,j].item())
                result = result[tmp]                                        # 判断属于2重类还是3重类
                if tmp == '0':
                    # 3重类
                    result = result[str(three_repeat_1_mask[i,j].item())]   # 判断其余两个轴向的关系
                    result = result[str(three_repeat_2_mask[i,j].item())]
                else:
                    # 2重类
                    result = result[str(repeat_2_class_mask[i,j].item())]   # 判断在哪个轴向还存在关系
                    result = result[str(two_repate_mask[i,j].item())]       # 判断在该轴向存在什么关系
                batch_list.append(result)
            direction_list.append(batch_list)
        return direction_list

    def calculate_adjacenct_matrix(self, data, dis_threshold=0.2, conf_threshold=0.7):
        centroid = data['centroid'] # 取3d中心点(相机坐标系)
        obj_conf = data['obj_conf'] # 取每个目标的置信度
        device = centroid.device
        B, N, _ = centroid.shape

        A = torch.ones((B, N, N)).to(device)                        # 初始化输出(邻接矩阵)
        eye_reverse = (torch.ones([N, N])-torch.eye(N)).to(device)  # 主对角线为0,其余为1,为了消去主对角线的自身关系
        eye_reverse = eye_reverse.unsqueeze(0).repeat(B, 1, 1)      # 为每个batch复制

        centroid_V = centroid.unsqueeze(2).repeat(1,1,N,1)          # 行复制，每一行都是一样的, [B,N,3] --> [B,N,1,3] --> [B,N,N,3]
        centroid_H = centroid.unsqueeze(1).repeat(1,N,1,1)          # 列复制，每一列都是一样的, [B,N,3] --> [B,1,N,3] --> [B,N,N,3]
        dist = torch.norm(centroid_V - centroid_H,dim=-1)           # 2范数计算距离, [B,N,N,3] --> [B,N,N]
        A[2:,2:] = A[2:,2:] * (dist > dis_threshold).float()[2:,2:] # 距离筛选, 筛选时不动前两行以及前两列, 保证sub和obj与其余保留节点都存在关系

        obj_conf_V = obj_conf.unsqueeze(-1).repeat(1,1,N)           # 行复制，每一行都是一样的, [B,N] --> [B,N,1] --> [B,N,N]
        obj_conf_H = obj_conf.unsqueeze(1).repeat(1,N,1)            # 列复制，每一列都是一样的, [B,N] --> [B,1,N] --> [B,N,N]
        obj_conf_V = (obj_conf_V >= conf_threshold).float()
        obj_conf_H = (obj_conf_H >= conf_threshold).float()

        A = A * obj_conf_V * obj_conf_H # 置信度筛选
        A = A * eye_reverse             # 抛弃自身连边
        # A[:,0,1], A[:,1,0]= 1, 1 # 确保sub和obj之间存在关系

        return A

if __name__ == 'main':
    model = Model()
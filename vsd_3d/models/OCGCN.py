import torch
import torch.nn as nn
from vsd_3d.models import FFN
import math

class OcGCN(nn.Module):
    def __init__(self,layer_num=3, d_prob=0.1):
        '''
    encoder部分
    '''
        super(OcGCN,self).__init__()
        self.pose_emb = nn.Embedding(8,768)

        # self.vison = FFN(in_feature = 2048,  # 这会使得损失值无法下降
        #                  middle_feature = 4096, 
        #                  out_feature = 768)
        self.vison_down = nn.Sequential(
            nn.Linear(2048,768),
            nn.LayerNorm(768)
        )
        self.vison = FFN(in_feature = 768, 
                         middle_feature = 3072, 
                         out_feature = 768)
        self.edge_up = nn.Sequential(
            nn.Linear(1,768),
            nn.LayerNorm(768)
        )
        self.edge = FFN(in_feature=768,middle_feature=3072,out_feature=768)

        self.Wb = nn.ModuleList()
        for _ in range(layer_num):
            self.Wb.append(
                        nn.Sequential(
                            nn.Linear(768,768),
                            nn.LayerNorm(768)
                                    )
                            ) # 这里是输出1还是输出768

        self.Wa = nn.ModuleList()
        for _ in range(layer_num):
            self.Wa.append(
                        nn.Sequential(
                            nn.Linear(768 * 3, 768),
                            nn.LayerNorm(768)
                                    )
                            )

        self.sigmoid = nn.Sigmoid()
        self.vision_layernorm = nn.LayerNorm(768)
        self.layernorm_list = nn.ModuleList([nn.LayerNorm(768)])
        self.vision_layernorm_list = nn.ModuleList([])
        self.wb_layernorm_list = nn.ModuleList([])
        # self.dropout = nn.Dropout(d_prob)
        for _ in range(layer_num):
            self.layernorm_list.append(nn.LayerNorm(768))
            self.vision_layernorm_list.append(nn.LayerNorm(768))
            self.wb_layernorm_list.append(nn.LayerNorm(768))        

    def forward(self,x, vis_feats, adjacency_matrix):
        vision, s_e = self.preprocess(x, vis_feats, adjacency_matrix) # vision [B, N, 768] s_e [B, N, N, 768]
        B, N, D = vision.shape
        adjacency_matrix_mask_float = adjacency_matrix.unsqueeze(-1).repeat(1,1,1,D).float() # [B, N, N] --> [B, N, N, 1] --> [B, N, N, D]
        adjacency_matrix_mask_reverse_float = 1 - adjacency_matrix_mask_float
        little_item = (torch.ones_like(adjacency_matrix_mask_float) * 1e-6) * adjacency_matrix_mask_reverse_float # 防止分母上为0导致结果为nan的情况
        
        # vision = self.laynorm(vision)
        st_v = self.layernorm_list[0](vision[:,0] + vision[:,1])            # sub和obj的视觉特征融合
        st_v = st_v.view(B,1,1,D).repeat(1,N,N,1)   # [B, D] --> [B, 1, 1, D] --> [B, N, N, D]
        
        # vision_res = vision
        for idx_layer in range(len(self.Wa)): # 公式2
            sj_v_ = vision.unsqueeze(1).repeat(1,N,1,1)     # [B, N, D] --> [B, 1, N, D] --> [B, N, N, D]
            a = self.wb_layernorm_list[idx_layer](sj_v_+st_v)
            a = a.view(B, N*N, D)                           # 每个batch展平，送入网络处理 [B, N, N, D] --> [B, N*N, D]
            a = self.Wb[idx_layer](a).view(B, N, N, D)      # 网络处理完后，形状恢复  [B, N*N, D] --> [B, N, N, D]
            a = torch.exp(a)

            # st_j_sum = vision.unsqueeze(1).repeat(1,N,1,1)  # [B, N, D] --> [B, 1, N, D] --> [B, N, N, D]
            # b = self.b_layernorm_list[idx_layer](st_j_sum+st_v)
            # b = b.view(B, N*N, D)                           # 每个batch展平，送入网络处理 [B, N, N, D] --> [B, N*N, D]
            # b = self.Wb[idx_layer](b).view(B, N, N, D)      # 网络处理完后，形状恢复  [B, N*N, D] --> [B, N, N, D]
            # b = torch.exp(b)
            b = a * adjacency_matrix_mask_float             # 只对存在关系的连边保持梯度
            b = torch.sum(b, dim=2, keepdim=True)           # [B, N, N, D] --> [B, N, 1, D]
            b = b.repeat(1,1,N,1) + little_item             # [B, N, 1, D] --> [B, N, N, D]

            gama = (a / b) * adjacency_matrix_mask_float    # 只保留存在关系连边的权重，其余置0

            si_v = vision.unsqueeze(2).repeat(1,1,N,1)                  # [B, N, D] --> [B, N, 1, D] --> [B, N, N, D]
            vision_cat = torch.cat([si_v, s_e, st_v],dim=-1)            # [B, N, N, 3*D]
            vision_cat = vision_cat.view(B, N*N, 3*D)                   # 每个batch展平，送入网络处理 [B, N, N, 3*D] --> [B, N*N, 3*D]
            vision = self.Wa[idx_layer](vision_cat).view(B, N, N, -1)   # 网络处理完后，形状恢复 [B, N*N, 3*D] --> [B, N*N, D] --> [B, N, N, D]
            vision = gama * vision                                      # 权重处理
            vision = torch.sum(vision,dim=2)                            # [B, N, N, D] --> [B, N, D]
            vision = self.sigmoid(vision)                               # 激活函数
            # vision = self.laynorm(vision)
            vision = self.vision_layernorm_list[idx_layer](vision)
            
            if idx_layer > 0:
                st_v = st_v_tmp
            st_v_tmp = self.layernorm_list[idx_layer+1](vision[:,0] + vision[:,1])                    # 使st_v作为上层视觉特征
            st_v_tmp = st_v_tmp.view(B,1,1,D).repeat(1,N,N,1)       # sj_v_作为本层视觉特征
            
        return vision, s_e

    def preprocess(self, data, vision, adjacency_matrix):
        centroid = data['centroid']
        B, N, _ = centroid.shape

        pose_index = self.oritation2int(data['basis'], data['coeffs'], self.pose_emb.weight.shape[0]) # 根据物体的姿态和大小确定方向朝向
        pose = self.pose_emb(pose_index.int())
        vision = self.vison_down(vision) # 2048 --> 768
        vision = self.vision_layernorm(vision + pose)
        vision = self.vison(vision)
        D = vision.shape[-1]
        adjacency_matrix_mask_float = adjacency_matrix.unsqueeze(-1).repeat(1,1,1,D) # [B, N, N] --> [B, N ,N ,1] --> [B, N, N, D]

        # 计算两两夹角余弦值
        # A·B = |A||B|cos(radian)
        local_V = centroid.unsqueeze(2).repeat(1,1,N,1) # [B, N, 3] --> [B, N, 1, 3] --> [B, N, N, 3]
        local_H = centroid.unsqueeze(1).repeat(1,N,1,1) # [B, N, 3] --> [B, 1, N, 3] --> [B, N, N, 3]
        # 计算点积
        dot_product = torch.sum(local_V * local_H, dim=-1)
        # 计算向量模长
        norm_local_V = torch.norm(local_V,dim=-1)
        norm_local_H = torch.norm(local_H,dim=-1)
        # 计算夹角的余弦值
        # radian = arccos[(A·B)/(|A||B|)]
        s_e = dot_product / (norm_local_V * norm_local_H)

        s_e = self.edge_up(s_e.view(B, N*N, 1))
        s_e = self.edge(s_e)                                # 每个batch展平送入网络处理 [B, N, N] --> [B, N*N, 1] --> [B, N*N, D]
        s_e = s_e.view(B, N, N, -1) * adjacency_matrix_mask_float    # 网络处理完恢复尺寸 [B, N*N, D] --> [B, N, N, D] 并仅使存在关系的边保持梯度
        return vision, s_e
    
    def oritation2int(self, oritation,size, num_area):
        '''
        对平面进行分区，根据方向朝向，返回指定的位置索引。
        '''
        B,N = oritation.shape[:2]
        one_step = 360. / num_area
        size = size.unsqueeze(-2).repeat(1,1,3,1)           # [B, N, 3] --> [B, N, 3, 3]
        oritation = oritation * size                        # 利用尺寸放缩，[B,N,3,3] * [B,N,3,3] 
        oritation = oritation[:,:,0] + oritation[:,:,-1]    # 取两方向向量和作为方向判据
        # 与相机坐标系下的水平轴方向计算角度
        # A·B = |A||B|cos(radian)
        camera_x_vector = torch.tensor([1.,0.,0.]).to(oritation.device)
        camera_x_vector = camera_x_vector.view(1,1,-1).repeat(B,N,1) # [3] --> [1, 1, 3] --> [B, N, 3]
        dot_product = torch.sum(oritation * camera_x_vector, dim=-1) # 计算点积

        # 计算向量模长
        norm_vector1 = torch.norm(oritation,dim=-1)
        norm_vector2 = torch.norm(camera_x_vector,dim=-1)

        # 计算夹角的余弦值
        # radian = arccos[(A·B)/(|A||B|)]
        cosine = dot_product / (norm_vector1 * norm_vector2)

        # 使用 arccos 函数计算夹角的弧度
        angle_radians = torch.acos(cosine)

        # 将弧度转换为角度
        angle_degrees = angle_radians * (180.0 / math.pi)

        # 确保角度在0到360度之间
        mask = torch.cross(oritation, camera_x_vector)[:,:,2]<0
        angle_degrees[mask] = 360 - angle_degrees[mask]
        index = angle_degrees // one_step # 确定方位属于哪个分区
        return index
    
    def forward_(self,x, vis_feats, adjacency_matrix): # 节省显存，牺牲速度
        adjacency_matrix_mask = adjacency_matrix == 1
        adjacency_matrix_mask_reverse = adjacency_matrix == 0
        vision, s_e = self.preprocess(x, vis_feats, adjacency_matrix_mask) # vision [B,36,768] s_e [B,36,36,768]
        B, N, D = vision.shape
        adjacency_matrix_mask_float = adjacency_matrix_mask.unsqueeze(-1).repeat(1,1,1,D).float()
        adjacency_matrix_mask_reverse_float = adjacency_matrix_mask_reverse.unsqueeze(-1).repeat(1,1,1,D).float()
        little_item = (torch.ones_like(adjacency_matrix_mask_float) * 1e-6) * adjacency_matrix_mask_reverse_float
        
        st_v = vision[:,0] + vision[:,1]
        st_v = st_v.view(B,1,1,D).repeat(1,N,N,1)

        for idx_layer in range(len(self.Wa)): # 公式2
            

            sj_v_ = vision.unsqueeze(1).repeat(1,N,1,1) # [B,36,768] --> [B,1,36,768] --> [B,36,36,768]
            a = sj_v_  + st_v
            a_tmp = a[adjacency_matrix_mask] # 只让存在边关系的节点进行方向传播
            a = self.Wb[idx_layer](a_tmp)
            a[adjacency_matrix_mask] = a_tmp
            a = torch.exp(a)
            a = a * adjacency_matrix_mask_float
            # a[adjacency_matrix_mask_reverse] = 0

            st_j_sum = vision.unsqueeze(1).repeat(1,N,1,1)
            b = st_j_sum + st_v
            b_tmp = b[adjacency_matrix_mask]
            b = self.Wb[idx_layer](b_tmp)
            b[adjacency_matrix_mask] = b_tmp
            b = torch.exp(b)
            b = b * adjacency_matrix_mask_float
            b = torch.sum(b, dim=2, keepdim=True)
            b = b.repeat(1,1,N,1) + little_item

            gama = (a / b) * adjacency_matrix_mask_float

            si_v = vision.unsqueeze(2).repeat(1,1,N,1)
            vision = vision.unsqueeze(2).repeat(1,1,N,1)
            vision_cat = torch.cat([si_v, s_e, st_v],dim=-1) # [B, N, N, 3*D]
            vision_tmp = vision_cat[adjacency_matrix_mask]
            vision_tmp = self.Wa[idx_layer](vision_tmp)
            vision[adjacency_matrix_mask] = vision_tmp
            vision = gama * vision
            vision = torch.sum(vision,dim=2) # [B,N,D]
            vision = self.sigmoid(vision)
            
            if idx_layer == 0:
                st_v_tmp = vision[:,0] + vision[:,1]
                st_v_tmp = st_v_tmp.view(B,1,1,D).repeat(1,N,N,1)
            else:
                st_v = st_v_tmp
                st_v_tmp = vision[:,0] + vision[:,1]
                st_v_tmp = st_v_tmp.view(B,1,1,D).repeat(1,N,N,1)


        return vision, s_e
    def preprocess_(self, data, vision, adjacency_matrix_mask): # 节省显存,牺牲速度
        obj_conf = data['obj_conf']
        centroid = data['centroid'] # 也要转换到相机坐标系下
        device = obj_conf.device
        B, N = obj_conf.shape

        pose_index = self.oritation2int(data['basis'], data['coeffs'], self.pose_emb.weight.shape[0])
        pose = self.pose_emb(pose_index.int())
        vision = self.vison(vision + pose) # 2048 --> 768
        # vision = self.vision_down_linear(vision) # 

        # TODO(将中心点转换到相机坐标系下，计算两两夹角余弦值) #这一步应该在dataset中处理 
        local_V = centroid.unsqueeze(2).repeat(1,1,N,1)
        local_H = centroid.unsqueeze(1).repeat(1,N,1,1)
        # 计算点积
        dot_product = torch.sum(local_V * local_H, dim=-1)
        # 计算向量长度
        norm_local_V = torch.norm(local_V,dim=-1)
        norm_local_H = torch.norm(local_H,dim=-1)
        # 计算夹角的余弦值
        s_e = dot_product / (norm_local_V * norm_local_H)
        s_e_result = torch.zeros([B,N,N,768]).to(device)

        s_e_tmp = s_e[adjacency_matrix_mask] # 只让存在边关系的节点进行反向传播
        s_e_res = self.edge(s_e.view(-1, 1))
        # s_e_result = s_e_res.view(B, N, N, -1) * adjacency_matrix_mask_float
        s_e_result[adjacency_matrix_mask] = s_e_res
        return vision, s_e_result
    
if __name__ == '__main__':
    model = OcGCN()
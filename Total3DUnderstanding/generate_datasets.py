import h5py
from scipy.io import loadmat
import os
import numpy as np
import json
mat_root = '/home/zhangjx/All_model/Total3DUnderstanding/save_mat_gt_path'
layout_mat_name = 'layout.mat'
bdb_3d_mat_name = 'bdb_3d.mat'
r_ex_mat_name = 'r_ex.mat'
basis_name = 'basis'
coeffs_name = 'coeffs'
centroid_name = 'centroid'
class_name = 'class'
vocab_path = '/home/zhangjx/All_model/genration_scene/VSD/datasets/objects_vocab.txt'
detection_root = '/home/zhangjx/All_model/Total3DUnderstanding/data_detection'
with open(vocab_path, 'r', encoding='utf-8') as f:
    VGSPCLASS = f.read().split('\n')
with h5py.File('/home/zhangjx/All_model/genration_scene/VSD/datasets/vsd_boxes36.h5','r') as vsd_data: # 原始2Dvsd数据集的.h5文件
    with h5py.File('/home/zhangjx/All_model/genration_scene/VSD/datasets/vsd3d_38.h5', 'a') as t3d_data:
        for idx, key in enumerate(vsd_data.keys()):
            detection_gt_path = os.path.join(detection_root, key, 'gt_detections.json')
            with open(detection_gt_path, 'r') as file:
                detections = json.load(file)
                ####### gt
                detections[0]['bbox'][0],detections[0]['bbox'][1],detections[0]['bbox'][2],detections[0]['bbox'][3] = detections[0]['bbox'][2],detections[0]['bbox'][0],detections[0]['bbox'][3],detections[0]['bbox'][1]
                detections[1]['bbox'][0],detections[1]['bbox'][1],detections[1]['bbox'][2],detections[1]['bbox'][3] = detections[1]['bbox'][2],detections[1]['bbox'][0],detections[1]['bbox'][3],detections[1]['bbox'][1]
                detections_gt = detections
            mat_one_path = os.path.join(mat_root,key)
            bdb_3d_path = os.path.join(mat_one_path,bdb_3d_mat_name)
            layout_path = os.path.join(mat_one_path,layout_mat_name)
            r_ex_path = os.path.join(mat_one_path,r_ex_mat_name)
            bdb_3d = loadmat(bdb_3d_path)
            layout = loadmat(layout_path)
            r_ex = loadmat(r_ex_path)
            basis_all = []
            coeffs_all = []
            centroid_all = []
            class_name_all = []
            for t3d_one in bdb_3d['bdb'][0]:
                basis = t3d_one[0][0][0]
                coeffs = t3d_one[0][0][1]
                centroid = t3d_one[0][0][2]
                class_name = t3d_one[0][0][3]

                basis_all.append(basis)
                coeffs_all.append(coeffs)
                centroid_all.append(centroid)
                class_name_all.append(str(class_name)[2:-2])
            basis_all = np.array(basis_all)
            coeffs_all = np.array(coeffs_all)
            centroid_all = np.array(centroid_all)

            box = vsd_data[f'{key}/boxes'][()]
            box = np.insert(box,0,np.array(detections[1]['bbox']),axis=0)
            box = np.insert(box,0,np.array(detections[0]['bbox']),axis=0)
            feature = np.load('/home/zhangjx/All_model/Total3DUnderstanding/feature_file/'+key+'/feature.npy')
            class_name_all = np.array(class_name_all,dtype='object')
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'object'+'/'+basis_name,data=basis_all)
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'object'+'/'+coeffs_name,data=coeffs_all)
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'object'+'/'+centroid_name,data=centroid_all)
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'object'+'/'+'class_name',data=class_name_all)
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'object'+'/'+'mask_ture_class',data=bdb_3d['mask_ture_class'])
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'r_ex',data=r_ex['cam_R'])
            t3d_data.create_dataset(key+'/'+'3d'+'/'+'layout',data=layout['layout'])
            t3d_data.create_dataset(f'{key}/features',data=feature)
            t3d_data.create_dataset(f'{key}/boxes',data=box)

            t3d_data.create_dataset(f'{key}/attr_conf',data=vsd_data[f'{key}/attr_conf'][()])
            t3d_data.create_dataset(f'{key}/attr_id',data=vsd_data[f'{key}/attr_id'][()])
            t3d_data.create_dataset(f'{key}/img_h',data=vsd_data[f'{key}/img_h'][()])
            t3d_data.create_dataset(f'{key}/img_w',data=vsd_data[f'{key}/img_w'][()])
            t3d_data.create_dataset(f'{key}/obj_conf',data=vsd_data[f'{key}/obj_conf'][()])
            t3d_data.create_dataset(f'{key}/obj_id',data=vsd_data[f'{key}/obj_id'][()])
            print(idx)
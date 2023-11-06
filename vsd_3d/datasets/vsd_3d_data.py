import torch
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json

class VSD_3d_dataset(Dataset):
    '''
    VSD3D数据集，在原有VSD数据集的.h文件上进行扩充，添加了经过Total3D处理的数据。
    旋转量basis，位置量centroid，尺寸量coeffs
    '''
    def __init__(self, args, split='train'):
        super(VSD_3d_dataset,self).__init__()
        self.args = args
        self.mode = split
        h5_path = '/home/zhangjx/All_model/genration_scene/3DVSD/data/vsd_boxes36.h5'
        dataset_dir = '/home/zhangjx/All_model/genration_scene/3DVSD/data'
        data_info_path = os.path.join(dataset_dir,f'VSDv1/{split}.json')
        with open(data_info_path) as f:
            dataset = json.load(f)

        n_images = 0
        data = []
        for datum in dataset:
            img_id = datum['img_id'].replace('.jpg', "")
            img_id = img_id.replace('.png', "")
            if self.mode == 'train':
                for d in datum['captions']:
                    new_datum = {
                        'img_id': img_id,
                        'sent': d.strip(),
                        'subject_and_objects': [[triple['s'], triple['p'], triple['o']] for triple in datum['triple_list']],
                        "predicate": [triple['p'] for triple in datum['triple_list']],
                        'targets': [caption.strip() for caption in datum['captions']],
                        'is_train': True
                    }
                    data.append(new_datum)
            else:
                new_datum = {
                    'img_id': img_id,
                    # 'subject_and_objects': [(triple['s'], triple['o']) for triple in datum['triple_list']],
                    'subject_and_objects': [[triple['s'], triple['p'], triple['o']] for triple in datum['triple_list']],
                    'targets': [caption.strip() for caption in datum['captions']],
                    'is_train': False
                }
                data.append(new_datum)
            n_images += 1
                
        # if self.verbose:
        #     print(f"{self.source} has f'{n_images}' images")
        #     print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        # self.rank = rank
        # if self.topk > 0:
        #     data = data[:self.topk]
        #     if self.verbose:
        #         print(f"Use only {self.topk} data")

        self.data = data

        self.source_to_h5 = h5py.File(h5_path,'r')

    def __getitem__(self, idx):
        out_dict = {}
        datum = self.data[idx]
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            # Normalize the boxes (to 0 ~ 1)
            # try:
            img_h = self.source_to_h5[f'{img_id}/img_h'][()]
            img_w = self.source_to_h5[f'{img_id}/img_w'][()]
            boxes = self.source_to_h5[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            # except:
            #     print(img_id)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            self.source_to_h5[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            # if self.args.n_boxes == 100:
            #     assert n_boxes == 100
            #     assert len(feats) == 100
            #     assert len(boxes) == 100

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vision'] = feats
            # r_ex = self.source_to_h5[f'{img_id}/3d/r_ex'][()]
            # out_dict['r_ex'] = r_ex
            # r_ex = r_ex.reshape(1,3,3).repeat(n_boxes, axis=0)
            basis = self.source_to_h5[f'{img_id}/3d/object/basis'][()]
            # out_dict['basis'] = np.matmul(r_ex, basis)
            out_dict['basis'] = basis
            out_dict['coeffs'] = self.source_to_h5[f'{img_id}/3d/object/coeffs'][()].reshape([n_boxes,3])
            centroid = self.source_to_h5[f'{img_id}/3d/object/centroid'][()].reshape([n_boxes,3,1])
            # out_dict['centroid'] = np.matmul(r_ex, centroid)
            out_dict['centroid'] = centroid.reshape(36,3)
            obj_conf = self.source_to_h5[f'{img_id}/obj_conf'][()][:-2]
            obj_conf = np.insert(obj_conf,0,1)
            obj_conf = np.insert(obj_conf,0,1)
            out_dict['obj_conf'] = obj_conf
        return out_dict

    def __len__(self):
        return len(self.data)
    

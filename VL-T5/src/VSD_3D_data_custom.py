from torch.utils.data import Dataset
import json
import os
from copy import deepcopy
import cv2
from PIL import Image
import numpy as np
class VSD_3D_Custom(Dataset):
    def __init__(self, img_path, json_path):
        super().__init__()
        self.cam_K = np.array([[529.5,   0. , 365. ],
                                [  0. , 529.5, 265. ],
                                [  0. ,   0. ,   1. ]])
        self.img_path = img_path
        self.json_path = json_path
        img_name_list = os.listdir(img_path)
        img_name_list_copy = deepcopy(img_name_list)
        for img_name in img_name_list_copy:
            suffix = img_name.split('.')[-1]
            if suffix == 'png':
                pass
            elif suffix == 'jpg':
                pass
            elif suffix == 'jpeg':
                pass
            else:
                img_name_list.remove(img_name)

        json_name_list = os.listdir(json_path)
        json_name_list_copy = deepcopy(json_name_list)
        for json_name in json_name_list_copy:
            suffix = json_name.split('.')[-1]
            if suffix != 'json':
                json_name_list.remove(json_name)
        
        self.img_name_list = img_name_list
        self.json_name_list = json_name_list
    
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        name = '.'.join(img_name.split('.')[:-1])
        json_path = os.path.join(self.json_path,name+'.json')
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        json_content = json.load(open(json_path))
        obj = json_content['shapes'][0]
        sub = json_content['shapes'][1]

        boxes = [[*obj['points'][0],*obj['points'][1]],
               [*sub['points'][0],*sub['points'][1]]
        ]
        boxes = np.array(boxes,dtype=np.float32)
        dic = {}
        dic['img'] = img
        dic['image'] = image
        dic['boxes'] = boxes
        dic['class'] = np.array([obj['label'], sub['label']],dtype='object')
        dic['cam_K'] = self.cam_K
        dic['name'] = name
        return dic

    def collate_fn(self, batch):
        return batch[0]
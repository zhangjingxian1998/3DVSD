# TODO 更改数据集形式
# 更改输入的提示模板 模板由两部分构成，
# 1、两个类别 <TGT> table <TGT> sofa # 两个target应该是指定的， 关系是预测的
# 2、空间关系 <OBJ> table <REL> near <OBJ> sofa  <OBJ> sofa <REL> left <OBJ> bed
# 最终提示模板 <TGT> table <TGT> sofa <SEP> <OBJ> table <REL> near <OBJ> sofa <OBJ> sofa <REL> left <OBJ> bed
# 最终提示模板 <table,sofa> <SEP> <table,near,sofa> <sofa,left,bed>

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import h5py
import torch
import numpy as np

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast
import copy

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('data/').resolve()
vrd_dir = dataset_dir.joinpath('sp3000')
vg_dir = dataset_dir.joinpath('vg')
vg_feature_dir = vg_dir.joinpath('features')
vrd_img_dir = vrd_dir.joinpath('images/')
vrd_feature_dir = vrd_dir.joinpath('features')

predicate = ["on", "to the left of", "under", "behind", "to the right of", "in", "next to", "in front of", "above"]
predicate_map = {p: i for i, p in enumerate(predicate)}
synonyms = {
    'on':['upon', 'over'],
    'to the left of':['to the left of'],
    'under':['below', 'beneath', 'underneath'],
    'behind':['after', 'at the back of', 'back'],
    'to the right of':['to the right of'],
    'in':['in'],
    'next to':['next to'],
    'in the front of':['ahead of'],
    'above':['upon', 'over']
}

class VSD_3D_FineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vrd_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        # data_info_path = dataset_dir.joinpath(f'sp3000/{split}.json')
        data_info_path = dataset_dir.joinpath(f'{args.data}/{split}.json')
        # data_info_path = dataset_dir.joinpath(f'VSDv2/{split}.json')

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
                        # "so_bbox": [[triple['s_bbox'], triple['o_bbox']] for triple in datum['triple_list']],
                        'is_train': True
                    }
                    data.append(new_datum)
            else:
                for d in datum['captions']:
                    new_datum = {
                        'img_id': img_id,
                        'sent': d.strip(),
                        # 'subject_and_objects': [(triple['s'], triple['o']) for triple in datum['triple_list']],
                        'subject_and_objects': [[triple['s'], triple['p'], triple['o']] for triple in datum['triple_list']],
                        'targets': [caption.strip() for caption in datum['captions']],
                        # "so_bbox": [[triple['s_bbox'], triple['o_bbox']] for triple in datum['triple_list']],
                        'is_train': False
                    }
                data.append(new_datum)
                n_images += 1
                
        if self.verbose:
            print(f"{self.source} has f'{n_images}' images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        if self.args.max_n_boxes == 36:
            self.source_to_h5 = dataset_dir.joinpath('vsd_3d_boxes38.h5')
            if isinstance(self.source_to_h5, Path):
                self.source_to_h5 = h5py.File(self.source_to_h5, 'r')
        
        ################################################################################
        # self.prompt_template = np.array(['<TGT>', 'sub', '<TGT>', 'obj', 
        #                         '[SEP]', 
        #                         '<OBJ>', 'sub', '<REL>', 'rel', '<OBJ>', 'middle',
        #                         '<OBJ>', 'middle', '<REL>', 'rel', '<OBJ>', 'obj']).astype(np.object_)
        # self.prompt_template_replace_all_id = [1, 3, 6, 8, 10, 12, 14, 16]
        # self.prompt_template_replace_sub_obj_id = [1, 3, 6, 16] # 替换sub和obj处的词
        # self.prompt_template_replace_middle_id = [10, 12] # 替换子图中间位置的词
        # self.prompt_template_replace_rel_id = [8, 14] # 替换关系词

        # self.prompt_template_VL_pretrain = np.array(['<TGT>', 'sub', '<TGT>', 'obj', 
        #                                     '[SEP]', 
        #                                     '<OBJ>', 'sub', '<REL>', 'rel', '<OBJ>', 'obj']).astype(np.object_)
        # self.prompt_template_VL_pretrain_replace_id = [1, 3, 6, 8, 10] 
        # self.prompt_template_VL_pretrain_replace_sub_obj_id = [1, 3, 6, 10] # 替换sub和obj处的词
        # self.prompt_template_VL_pretrain_replace_rel_id = [8] # 替换关系词

        # self.prompt_template_VL_pretrain = np.array(['<OBJ>', 'sub', '<REL>', 'rel', '<OBJ>', 'obj']).astype(np.object_)
        # self.prompt_template_VL_pretrain_replace_id = [1, 3, 5] 
        # self.prompt_template_VL_pretrain_replace_sub_obj_id = [1, 5] # 替换sub和obj处的词
        # self.prompt_template_VL_pretrain_replace_rel_id = [3] # 替换关系词
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 对VL模型预训练时, 能给到的视觉提示是前两个, 即sub和obj
        f = self.source_to_h5
        if self.args.VL_pretrain:
            out_dict = {}
            out_dict_3dvsd = {}
            out_dict['args'] = self.args

            datum = self.data[idx]

            if self.args.use_vision:
                img_id = datum['img_id']
                out_dict['img_id'] = img_id

                # Normalize the boxes (to 0 ~ 1)
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = f[f'{img_id}/boxes'][()][:2]  # (x1, y1, x2, y2)

                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                boxes[boxes>1] = 1
                boxes[boxes<0] = 0 # 有些框越界, 进行调整, 防止越界

                boxes = torch.from_numpy(boxes)

                boxes.clamp_(min=0.0, max=1.0)

                n_boxes = len(boxes)
                
                boxes_center_w = (boxes[:,2:3] + boxes[:,0:1]) / 2
                boxes_center_h = (boxes[:,-1:] + boxes[:,1:2]) / 2
                boxes_center_3d = torch.cat([torch.zeros(n_boxes,1), boxes_center_w, boxes_center_h], dim=-1) # 深度方向为0, 其余取自身的2D location
                # feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                feats = f[f'{img_id}/features'][()][:2]
                feats = torch.from_numpy(feats)

                if self.args.n_boxes == 100:
                    assert n_boxes == 100
                    assert len(feats) == 100
                    assert len(boxes) == 100

                out_dict['n_boxes'] = n_boxes
                boxes = boxes[:n_boxes]
                feats = feats[:n_boxes]
                out_dict['boxes'] = boxes
                out_dict['vis_feats'] = feats
            ##### 3dvsd #####
            r_ex = f[f'{img_id}/3d/r_ex'][()]
            out_dict_3dvsd['r_ex'] = r_ex # 模型估计的相机外参矩阵
            centroid = f[f'{img_id}/3d/object/centroid'][()][:2] # 每个物体的中心坐标(世界坐标系)
            out_dict_3dvsd['class_name'] = f[f'{img_id}/3d/object/class_name'][()][:2] # 物体的类别名称(1600类)
            out_dict_3dvsd['mask_ture_class'] = f[f'{img_id}/3d/object/mask_ture_class'][()][0][:2] # 物体类别是否与nyu40类对应，对应的物体处为1
            out_dict_3dvsd['centroid'] = np.matmul(r_ex.reshape(1,3,3).repeat(n_boxes,axis=0), centroid.reshape(-1,3,1)).reshape(n_boxes,3) # 位置转换到相机坐标系
            out_dict_3dvsd['boxes_center_3d'] = boxes_center_3d

            # 不能直接在这里转input_id，因为子图没有，没有办法生成
            ###########   TEXT   ####################
            subject_and_object = datum['subject_and_objects'][0] # 取sub和obj
            # sub = subject_and_object[0]
            # rel_gt = subject_and_object[1]
            # obj = subject_and_object[2]
            # replace_sub_obj = [sub, obj]
            # # 将模板中的sub和obj位置替换为标签
            # input_text = copy.deepcopy(self.prompt_template_VL_pretrain)
            # input_text[self.prompt_template_VL_pretrain_replace_sub_obj_id] = replace_sub_obj

            # 先不tolist(), 将所有位置替换完后再tolist()
            ##################################################################################
            
            # out_dict['input_text'] = input_text
            
            target_text = ','.join(subject_and_object)
            out_dict['pretrain_target'] = target_text
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(target_text, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(target_text, max_length=self.args.gen_max_length, truncation=True)
            out_dict['pretrain_target_id'] = torch.LongTensor(target_ids)
            out_dict['pretrain_target_id_length'] = len(target_ids)

            return {'out_dict':out_dict, 'out_dict_3dvsd':out_dict_3dvsd}

        else:
            out_dict = {}
            out_dict_3dvsd = {}
            out_dict['args'] = self.args

            datum = self.data[idx]

            ###### Image ######
            if self.args.use_vision:
                img_id = datum['img_id']
                # img_id = '60'
                out_dict['img_id'] = img_id

                # Normalize the boxes (to 0 ~ 1)
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                boxes[boxes>1] = 1
                boxes[boxes<0] = 0 # 有些框越界, 进行调整, 防止越界

                boxes = torch.from_numpy(boxes)

                boxes.clamp_(min=0.0, max=1.0)

                n_boxes = len(boxes)

                boxes_center_w = (boxes[:,2:3] + boxes[:,0:1]) / 2
                boxes_center_h = (boxes[:,-1:] + boxes[:,1:2]) / 2

                boxes_center_w_show = boxes_center_w * img_w
                boxes_center_h_show = boxes_center_h * img_h
                boxes_center_3d_show = torch.cat([torch.zeros(n_boxes,1), boxes_center_w_show, boxes_center_h_show], dim=-1)

                boxes_center_3d = torch.cat([torch.zeros(n_boxes,1), 1-boxes_center_h, boxes_center_w], dim=-1) # 深度方向为0, 其余取自身的2D location

                feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                f[f'{img_id}/features'].read_direct(feats)
                feats = torch.from_numpy(feats)

                if self.args.n_boxes == 100:
                    assert n_boxes == 100
                    assert len(feats) == 100
                    assert len(boxes) == 100

                # n_boxes = min(n_boxes, self.args.max_n_boxes)
                out_dict['n_boxes'] = n_boxes
                boxes = boxes[:n_boxes]
                feats = feats[:n_boxes]
                out_dict['boxes'] = boxes
                out_dict['vis_feats'] = feats

            ##### 3dvsd #####
            r_ex = f[f'{img_id}/3d/r_ex'][()]
            out_dict_3dvsd['r_ex'] = r_ex # 模型估计的相机外参矩阵
            basis = f[f'{img_id}/3d/object/basis'][()] # 每个物体的姿态(世界坐标系)
            centroid = f[f'{img_id}/3d/object/centroid'][()] # 每个物体的中心坐标(世界坐标系)
            out_dict_3dvsd['coeffs'] = f[f'{img_id}/3d/object/coeffs'][()].reshape(n_boxes,3) # 每个物体的尺寸
            out_dict_3dvsd['class_name'] = f[f'{img_id}/3d/object/class_name'][()] # 物体的类别名称(1600类)
            out_dict_3dvsd['mask_ture_class'] = f[f'{img_id}/3d/object/mask_ture_class'][()].reshape(n_boxes) # 物体类别是否与nyu40类对应，对应的物体处为1
            out_dict_3dvsd['basis'] = np.matmul(r_ex.reshape(1,3,3).repeat(n_boxes,axis=0), basis) # 姿态转换到相机坐标系
            # out_dict_3dvsd['basis'] = basis # 姿态不应直接变到相机坐标系
            out_dict_3dvsd['centroid'] = np.matmul(r_ex.reshape(1,3,3).repeat(n_boxes,axis=0), centroid.reshape(-1,3,1)).reshape(n_boxes,3) # 位置转换到相机坐标系
            obj_conf = self.source_to_h5[f'{img_id}/obj_conf'][()]
            obj_conf = np.insert(obj_conf,0,1)
            obj_conf = np.insert(obj_conf,0,1)
            out_dict_3dvsd['obj_conf'] = obj_conf
            out_dict_3dvsd['boxes_center_3d'] = boxes_center_3d
            out_dict_3dvsd['boxes_center_3d_show'] = boxes_center_3d_show

            # 不能直接在这里转input_id，因为子图没有，没有办法生成
            ###########   TEXT   ###################################################################### TODO(zhangjignxian) 输入词替换
            # subject_and_object = datum['subject_and_objects'][0] # 取sub和obj
            # sub = subject_and_object[0]
            # rel_gt = subject_and_object[1]
            # obj = subject_and_object[2]
            # replace_sub_obj = [sub, obj, sub, obj]
            # # 将模板中的sub和obj位置替换为标签
            # if self.args.VL_pretrain:
            #     input_text = copy.deepcopy(self.prompt_template_VL_pretrain)

            #     input_text[self.prompt_template_VL_pretrain_replace_sub_obj_id] = [sub, obj, sub, obj]
            #     # input_text = self.prompt_template_VL_pretrain
            # else:
            #     input_text = copy.deepcopy(self.prompt_template)
            #     input_text[self.prompt_template_replace_sub_obj_id] = replace_sub_obj
                # input_text = self.prompt_template
            # 先不tolist(), 将所有位置替换完后再tolist()
            ##################################################################################
            
            # out_dict['input_text'] = input_text

            # out_dict['input_ids'] = torch.LongTensor(input_ids)
            # out_dict['input_length'] = len(input_ids)
            
            # if datum['is_train']:
            sent = datum['sent'].strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)

            out_dict['sent'] = sent
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            if 'targets' in datum:
                out_dict['targets'] = datum['targets']

            return {'out_dict':out_dict, 'out_dict_3dvsd':out_dict_3dvsd}

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)
        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        sentences = []
        if self.args.VL_pretrain:
            if self.args.use_vision:
                V_L = max(entry['out_dict']['n_boxes'] for entry in batch)
                feat_dim = batch[0]['out_dict']['vis_feats'].shape[-1]

                T_W_L = max(entry['out_dict']['pretrain_target_id_length'] for entry in batch)
                target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id


                boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
                vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
                vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

            for i, entry in enumerate(batch):
                if self.args.use_vision:
                    n_boxes = entry['out_dict']['n_boxes']
                    boxes[i, :n_boxes] = entry['out_dict']['boxes']
                    vis_feats[i, :n_boxes] = entry['out_dict']['vis_feats']
                    vis_attention_mask[i, :n_boxes] = 1
                    img_ids.append(entry['out_dict']['img_id'])
                    target_ids[i, :entry['out_dict']['pretrain_target_id_length']] = entry['out_dict']['pretrain_target_id']

                # if 'input_text' in entry['out_dict']:
                #     input_text.append(entry['out_dict']['input_text'])

                targets.append(entry['out_dict']['pretrain_target'])
            
            if self.args.use_vision:
                batch_entry['boxes'] = boxes
                # batch_entry['vis_feats'] = vis_feats
                batch_entry['vis_attention_mask'] = vis_attention_mask
                batch_entry['img_id'] = img_ids

            # batch_entry['input_text'] = np.array(input_text)

            batch_entry['targets'] = targets

            batch_entry['task'] = 'caption'

            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

            # vsd_3d 需要输入经过total3d处理后的 
            # 外参矩阵:r_ex 
            # 目标的三维框姿态:basis
            # 目标的三维坐标:centroid
            # 目标的尺寸大小:coeffs
            # 物体目标是否与total3d类别对应:mask_ture_class
            # 类别名称:class_name
            # 每个类别的置信度:obj_conf
            batch_entry_3d = {}

            # 初始化
            N = batch[0]['out_dict_3dvsd']['centroid'].shape[0]
            r_ex = torch.zeros(B, 3, 3, dtype=torch.float)
            centroid = torch.zeros(B, N, 3, dtype=torch.float)
            mask_ture_class = torch.zeros(B, N, dtype=torch.float)
            boxes_center_3d = torch.zeros(B, N, 3, dtype=torch.float)
            class_name = []

            for i, entry in enumerate(batch):
                r_ex[i, : ] = torch.tensor(entry['out_dict_3dvsd']['r_ex'])
                centroid[i, : ] = torch.tensor(entry['out_dict_3dvsd']['centroid'])
                boxes_center_3d[i,:] = entry['out_dict_3dvsd']['boxes_center_3d']
                mask_ture_class[i, : ] = torch.tensor(entry['out_dict_3dvsd']['mask_ture_class'])
                name_list = str(b'[MASK]'.join(list(entry['out_dict_3dvsd']['class_name'])))[2:-1].split('[MASK]')
                class_name.append(name_list)

            batch_entry_3d['r_ex'] = r_ex # [B,3,3]
            batch_entry_3d['centroid'] = centroid # [B,N,3]
            batch_entry_3d['mask_ture_class'] = mask_ture_class # [B,N]
            batch_entry_3d['class_name'] = np.array(class_name).astype(np.object_)
            batch_entry_3d['boxes_center_3d'] = boxes_center_3d
            
            return {'batch_entry':batch_entry, 'batch_entry_3d':batch_entry_3d, 'vis_feats':vis_feats}

        else:
            if self.args.use_vision:
                V_L = max(entry['out_dict']['n_boxes'] for entry in batch)
                feat_dim = batch[0]['out_dict']['vis_feats'].shape[-1]

                boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
                vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
                vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

            if 'target_ids' in batch[0]['out_dict']:
                T_W_L = max(entry['out_dict']['target_length'] for entry in batch)
                target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

            for i, entry in enumerate(batch):
                if self.args.use_vision:
                    n_boxes = entry['out_dict']['n_boxes']
                    boxes[i, :n_boxes] = entry['out_dict']['boxes']
                    vis_feats[i, :n_boxes] = entry['out_dict']['vis_feats']
                    vis_attention_mask[i, :n_boxes] = 1
                    img_ids.append(entry['out_dict']['img_id'])

                if 'target_ids' in entry['out_dict']:
                    target_ids[i, :entry['out_dict']['target_length']] = entry['out_dict']['target_ids']

                # if 'input_text' in entry['out_dict']:
                #     input_text.append(entry['out_dict']['input_text'])

                sentences.append(entry['out_dict']['sent'])

                if 'targets' in entry['out_dict']:
                    targets.append(entry['out_dict']['targets'])
                


            # batch_entry['input_ids'] = input_ids
            if 'target_ids' in batch[0]['out_dict']:
                word_mask = target_ids != self.tokenizer.pad_token_id
                target_ids[~word_mask] = -100
                batch_entry['target_ids'] = target_ids
            
            # if 'target_relation_ids' in batch[0]['out_dict']:
            #     word_mask = target_relation_ids != -1
            #     target_relation_ids[~word_mask] = -100
            #     batch_entry['target_relation_ids'] = target_relation_ids

            if self.args.use_vision:
                batch_entry['boxes'] = boxes
                # batch_entry['vis_feats'] = vis_feats
                batch_entry['vis_attention_mask'] = vis_attention_mask
                batch_entry['img_id'] = img_ids

            # batch_entry['input_text'] = np.array(input_text)

            batch_entry['targets'] = targets

            batch_entry['sentences'] = sentences

            batch_entry['task'] = 'caption'

            # vsd_3d 需要输入经过total3d处理后的 
            # 外参矩阵:r_ex 
            # 目标的三维框姿态:basis
            # 目标的三维坐标:centroid
            # 目标的尺寸大小:coeffs
            # 物体目标是否与total3d类别对应:mask_ture_class
            # 类别名称:class_name
            # 每个类别的置信度:obj_conf
            batch_entry_3d = {}

            # 初始化
            N = batch[0]['out_dict_3dvsd']['coeffs'].shape[0]
            r_ex = torch.zeros(B, 3, 3, dtype=torch.float)
            basis = torch.zeros(B, N, 3, 3, dtype=torch.float)
            centroid = torch.zeros(B, N, 3, dtype=torch.float)
            coeffs = torch.zeros(B, N, 3, dtype=torch.float)
            mask_ture_class = torch.zeros(B, N, dtype=torch.float)
            obj_conf = torch.zeros(B, N, dtype=torch.float)
            boxes_center_3d = torch.zeros(B, N, 3, dtype=torch.float)
            boxes_center_3d_show = torch.zeros(B, N, 3, dtype=torch.float)
            class_name = []

            for i, entry in enumerate(batch):
                r_ex[i, : ] = torch.tensor(entry['out_dict_3dvsd']['r_ex'])
                basis[i, : ] = torch.tensor(entry['out_dict_3dvsd']['basis'])
                centroid[i, : ] = torch.tensor(entry['out_dict_3dvsd']['centroid'])
                boxes_center_3d[i,:] = entry['out_dict_3dvsd']['boxes_center_3d']
                boxes_center_3d_show[i,:] = entry['out_dict_3dvsd']['boxes_center_3d_show']
                coeffs[i, : ] = torch.tensor(entry['out_dict_3dvsd']['coeffs'])
                mask_ture_class[i, : ] = torch.tensor(entry['out_dict_3dvsd']['mask_ture_class'])
                obj_conf[i, :] = torch.tensor(entry['out_dict_3dvsd']['obj_conf'])
                name_list = str(b'[MASK]'.join(list(entry['out_dict_3dvsd']['class_name'])))[2:-1].split('[MASK]')
                class_name.append(name_list)
                pass

            batch_entry_3d['r_ex'] = r_ex # [B,3,3]
            batch_entry_3d['basis'] = basis # [B,N,3,3]
            batch_entry_3d['centroid'] = centroid # [B,N,3]
            batch_entry_3d['coeffs'] = coeffs # [B,N,3]
            batch_entry_3d['mask_ture_class'] = mask_ture_class # [B,N]
            batch_entry_3d['class_name'] = np.array(class_name).astype(np.object_)
            batch_entry_3d['obj_conf'] = obj_conf
            batch_entry_3d['boxes_center_3d'] = boxes_center_3d
            batch_entry_3d['boxes_center_3d_show'] = boxes_center_3d_show
            
            return {'batch_entry':batch_entry, 'batch_entry_3d':batch_entry_3d, 'vis_feats':vis_feats}

def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = VSD_3D_FineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
        # train_sampler = RandomNonreplacmentSampler(dataset, dataset.n_iter)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'vrd_caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"],verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results
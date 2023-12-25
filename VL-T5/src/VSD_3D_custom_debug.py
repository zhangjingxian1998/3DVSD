# 正式训练时的输入模板变化, 还要额外训练一个3DVSD的encoder, 模型需要学习的提示词增加一个<TGT>
# 与之对比的输出是描述性语句
import sys
import os
current_path = ('/').join((sys.path[0].split('/'))[:-3])
sys.path.append(os.path.join(current_path, '3DVSD'))
sys.path.append(os.path.join(current_path, '3DVSD/Total3DUnderstanding'))
sys.path.append(os.path.join(current_path, '3DVSD/faster_rcnn'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.backends.cudnn as cudnn

from pathlib import Path

from tqdm import tqdm
import torch
import logging

from param import parse_args

from utils import set_global_logging_level
from Total3DUnderstanding.net_utils.utils import load_model, CheckpointIO
from Total3DUnderstanding.utils.param import parse_args as total3d_parse_args
from Total3DUnderstanding.configs.config_utils import CONFIG, mount_external_config
from Total3DUnderstanding.net_utils.utils import initiate_environment
from Total3DUnderstanding.configs.data_config import Relation_Config, NYU40CLASSES
from Total3DUnderstanding.net_utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation

from vsd_3d.models import Model
from faster_rcnn.detectron2_extract import build_model, extract_custom
from time import time
from VSD_3D_data_custom import VSD_3D_Custom
from torchvision import transforms
import numpy as np
import math
from pandas import DataFrame
import collections
# set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent

predict_list = ["to the left of", "to the right of", "in front of","next to", "above","behind","under","on","in"]

from trainer_base import TrainerBase
map_dict = {
    'rock wall':'wall','wall':'wall','walls':'wall','stone wall':'wall','brick wall':'wall','wallpaper':'wall',
    'floor':'floor','floors':'floor','flooring':'floor','tile floor':'floor',
    'cabinets':'cabinet','cabinet':'cabinet',
    'bed':'bed','beds':'bed','bed frame':'bed',
    'beach chair':'chair','chair':'chair','armchair':'chair','lounge chair':'chair','office chair':'chair','chairs':'chair','wheelchair':'chair','bench':'chair','benches':'chair','park bench':'chair',
    'sofa':'sofa','couch':'sofa',
    'table':'table','coffee table':'table','end table':'table','picnic table':'table','tables':'table','tv stand':'table',
    'garage door':'door','door':'door','cabinet door':'door','oven door':'door','shower door':'door','doors':'door','door frame':'door','glass door':'door',
    'window':'window','front window':'window','windows':'window','side window':'window',
    'bookshelf':'bookshelf',
    'picture':'picture','pictures':'picture','picture frame':'picture',
    'counter':'counter',
    'blinds':'blinds',
    'desk':'desk',
    'shelves':'shelves',
    'curtain':'curtain','curtains':'curtain',
    'dresser':'dresser',
    'pillow':'pillow','pillows':'pillow','throw pillow':'pillow','pillow case':'pillow',
    'mirror':'mirror','side mirror':'mirror',
    'mat':'floor_mat',
    'clothes':'clothes','pants':'clothes','jacket':'clothes','shirt':'clothes',
    'ceiling':'ceiling',
    'books':'books',
    'refrigerator,fridge':'refridgerator',
    'television,tv':'television',
    'papers':'paper','paper':'paper',
    'towel':'towel','towels':'towel','hand towel':'towel',
    'shower curtain':'shower_curtain',
    'tissue box':'box','box':'box','cardboard box':'box','boxes':'box',
    'board':'whiteboard',
    'person':'person','people':'person','man':'person','woman':'person','policeman':'person','young man':'person',
    'nightstand,night stand':'night_stand',
    'toilet':'toilet',
    'sink':'sink','bathroom sink':'sink','sinks':'sink',
    'floor lamp':'lamp','lamp':'lamp','lamps':'lamp','street lamp':'lamp','table lamp':'lamp',
    'bathtub':'bathtub',
    'bag':'bag','bags':'bag','trash bag':'bag','handbag':'bag',
}
HEIGHT_PATCH = 256
WIDTH_PATCH = 256
data_transforms = transforms.Compose([
                    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
class Trainer(TrainerBase):
    def __init__(self, args, rcnn_model, total3d_model, vsd_3d_encoder, dataloader):
        super().__init__(
            args,)
        self.rcnn_model = rcnn_model
        self.total3d_model = total3d_model
        self.vsd_3d_encoder = vsd_3d_encoder
        # build 3dvsd decoder
        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from VSD_3D_model import VLT53DVSD, VLBart3DVSD

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT53DVSD
        elif 'bart' in args.backbone:
            model_class = VLBart3DVSD

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                additional_special_tokens.append('<TGT>')
                additional_special_tokens.append('<OBJ>')
                additional_special_tokens.append('<REL>')
                additional_special_tokens.append('<SEP>')
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer: # 可以观察一下更改tokenizer长度后是否还可以加载权重 # 新添加词汇后, 原有的权重编码被保留, 后加的词汇初始化编码为固定的, 似乎有什么东西在控制
            # self.model.resize_token_embeddings(self.tokenizer.vocab_size)
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        print(f'Model Launching at GPU {self.args.gpu}')
        
        start = time()
        self.rcnn_model.model = self.rcnn_model.model.to(args.gpu)
        self.total3d_model = self.total3d_model.to(args.gpu)
        self.vsd_3d_encoder = self.vsd_3d_encoder.to(args.gpu)
        self.model = self.model.to(args.gpu)
        print(f'It took {time() - start:.1f}s')
        self.dataloader = dataloader

    def predict_custom(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length
            answer = []
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
            for i, data_one in enumerate(loader):
                rcnn_output = extract_custom(data_one['img'], self.rcnn_model)
                extra_3d_input = self.extract_3d(data_one,rcnn_output)
                result_3d = self.total3d_model(extra_3d_input)
                result_3d = get_3d_result(result_3d, extra_3d_input)
                batch = generate_batch(rcnn_output, extra_3d_input, result_3d)
                batch['batch_entry']['img_id'] = data_one['name']
                r_G, text_prompt, _ = self.vsd_3d_encoder(self.args, batch)
                batch['batch_entry']['input_ids'] = self.text_process(batch, text_prompt)
                results = self.model.test_step(batch, r_G,**gen_kwargs)
                pred_ans = results['pred_ans']
                if self.args.visualize:
                    name = batch['batch_entry']['img_id']
                    true = batch['batch_entry']['sentences']
                    if not os.path.exists(self.args.save_result_path):
                        os.mkdir(self.args.save_result_path)
                    with open(f'{self.args.save_result_path}/{name}.txt', 'w') as t:
                        t.write('The model output: \n')
                        for ans in pred_ans:
                            t.write(ans)
                            t.write('\n')
                        t.close()
                answer.append(pred_ans)
                pbar.update(1)
        return answer 
    def extract_3d(self, batch, rcnn_output):
        boxes = dict()
        camera = dict()
        camera['K'] = batch['cam_K']
        class_code = np.concatenate([batch['class'],rcnn_output['class']])
        class_dict = {'class':class_code.tolist()}
        class_dict = DataFrame(data=class_dict)
        class_dict['new_class'] = class_dict['class'].map(map_dict)
        new_class = class_dict['new_class'].to_list()
        mask_ture_class = []
        for i in range(len(new_class)):
            if type(new_class[i]) != type('a'):
                new_class[i] = 'void'
                mask_ture_class.append(0)
            else:
                mask_ture_class.append(1)
        bdb2D_pos = []
        bdb2D_pos.extend(batch['boxes'].tolist())
        bdb2D_pos.extend(rcnn_output['boxes'].tolist())
        boxes['g_feature'] = get_g_features(bdb2D_pos)
        boxes['bdb2D_pos'] = np.array(bdb2D_pos)
        size_cls = []
        for i in new_class:
            size_cls.append(NYU40CLASSES.index(i))
        cls_codes = torch.zeros([len(size_cls), len(NYU40CLASSES)])
        cls_codes[range(len(size_cls)), size_cls] = 1 # one_hot
        boxes['size_cls'] = cls_codes

        patch = []
        for bdb in bdb2D_pos:
            img = batch['image'].crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = data_transforms(img)
            patch.append(img)
        boxes['patch'] = torch.stack(patch)
        image = data_transforms(batch['image'])
        data = collate_fn([{'image':image, 'boxes_batch':boxes, 'camera':camera}])
        image = data['image']
        K = data['camera']['K'].float()
        patch = data['boxes_batch']['patch']
        size_cls = data['boxes_batch']['size_cls'].float()
        g_features = data['boxes_batch']['g_feature'].float()
        split = data['obj_split']
        rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)

        bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float()
        input_data = {'image':image, 'K':K, 'patch':patch, 'g_features':g_features,
                  'size_cls':size_cls, 'split':split, 'rel_pair_counts':rel_pair_counts,
                    'bdb2D_pos':bdb2D_pos, 'class_code':class_code,'mask_ture_class':mask_ture_class}
        return input_data
    def text_process(self, batch, text_prompt):
        B = batch['vis_feats'].shape[0]
        input_ids = []
        for text in text_prompt:
            input_ids.append(self.tokenizer.encode(text, return_tensors='pt', max_length=self.args.max_text_length, truncation=True)[0])
        S_W_L = 0
        length = []
        for input_id in input_ids:
            if input_id.shape[0] > S_W_L:
                S_W_L = input_id.shape[0]
            length.append(input_id.shape[0])
        input_ids_tensor = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        for i in range(B):
            input_ids_tensor[i,:length[i]] = input_ids[i]
        return input_ids_tensor

    def test(self):
        best_path = os.path.join(self.args.output, 'BEST.pth')
        self.load_checkpoint(best_path)
        answer = self.predict_custom(self.dataloader)
        print(answer)
def generate_batch(rcnn_output, extra_3d_input, result_3d):
    img_h = rcnn_output['img_h']
    img_w = rcnn_output['img_w']
    boxes = extra_3d_input['bdb2D_pos']
    boxes[:, (0, 2)] /= img_w
    boxes[:, (1, 3)] /= img_h
    boxes[boxes>1] = 1
    boxes[boxes<0] = 0 # 有些框越界, 进行调整, 防止越界
    boxes.clamp_(min=0.0, max=1.0)
    n_boxes = len(boxes)

    boxes_center_w = (boxes[:,2:3] + boxes[:,0:1]) / 2
    boxes_center_h = (boxes[:,-1:] + boxes[:,1:2]) / 2

    boxes_center_w_show = boxes_center_w * img_w
    boxes_center_h_show = boxes_center_h * img_h
    boxes_center_3d_show = torch.cat([torch.zeros_like(boxes_center_w_show), boxes_center_w_show, boxes_center_h_show], dim=-1)

    boxes_center_3d = torch.cat([torch.zeros_like(boxes_center_h), 1-boxes_center_h, boxes_center_w], dim=-1) # 深度方向为0, 其余取自身的2D location
    batch_entry = {}
    batch_entry_3d = {}
    batch_entry_3d['r_ex'] = result_3d['cam_K']
    batch_entry_3d['basis'] = torch.matmul(result_3d['cam_K'].repeat(38,1,1),result_3d['basis']).unsqueeze(0)
    batch_entry_3d['centroid'] = torch.matmul(result_3d['cam_K'].repeat(38,1,1),result_3d['centroid']).squeeze(-1).unsqueeze(0)
    batch_entry_3d['coeffs'] = result_3d['coeffs'].unsqueeze(0)
    batch_entry_3d['mask_ture_class'] = torch.tensor(extra_3d_input['mask_ture_class']).unsqueeze(0)
    batch_entry_3d['class_name'] = extra_3d_input['class_code'].reshape(1,extra_3d_input['class_code'].shape[0])
    batch_entry_3d['obj_conf'] = torch.cat([torch.ones(2).to(rcnn_output['obj_conf'].device),rcnn_output['obj_conf']]).unsqueeze(0)
    batch_entry_3d['boxes_center_3d_show'] = boxes_center_3d_show.unsqueeze(0)
    batch_entry_3d['boxes_center_3d'] = boxes_center_3d.unsqueeze(0)

    batch_entry['boxes'] = boxes.unsqueeze(0)
    batch_entry['vis_attention_mask'] = torch.ones_like(batch_entry_3d['obj_conf'])
    batch_entry['sentences'] = 'None'
    return {'batch_entry':batch_entry,
            'batch_entry_3d':batch_entry_3d,
            'vis_feats':result_3d['feature_roi'].unsqueeze(0)}

def get_3d_result(est_data, data):
    # lo_bdb3D_out = get_layout_bdb_sunrgbd(cfg.bins_tensor, est_data['lo_ori_reg_result'],
    #                                       torch.argmax(est_data['lo_ori_cls_result'], 1),
    #                                       est_data['lo_centroid_result'],
    #                                       est_data['lo_coeffs_result'])
    # camera orientation for evaluation
    cam_R_out = get_rotation_matix_result(cfg.bins_tensor,
                                          torch.argmax(est_data['pitch_cls_result'], 1), est_data['pitch_reg_result'],
                                          torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'])

    # projected center
    P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                            (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
                            (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                            (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:,1]), 1)

    bdb3D_out_form, bdb3D_out = get_bdb_evaluation(cfg.bins_tensor,
                                                       torch.argmax(est_data['ori_cls_result'], 1),
                                                       est_data['ori_reg_result'],
                                                       torch.argmax(est_data['centroid_cls_result'], 1),
                                                       est_data['centroid_reg_result'],
                                                       data['size_cls'], est_data['size_reg_result'], P_result,
                                                       data['K'], cam_R_out, data['split'], return_bdb=True)
    result = {}
    result['cam_K'] = cam_R_out
    result['basis'] = bdb3D_out_form['basis']
    result['coeffs'] = bdb3D_out_form['coeffs']
    result['centroid'] =bdb3D_out_form['centroid']
    result['feature_roi'] = est_data['feature_roi']
    result['box_center'] = P_result
    return result
def get_g_features(bdb2D_pos):
    rel_cfg = Relation_Config()
    d_model = int(rel_cfg.d_g/4)
    n_objects = len(bdb2D_pos)
    g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                  ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                  math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                  math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                 for id1, loc1 in enumerate(bdb2D_pos)
                 for id2, loc2 in enumerate(bdb2D_pos)]
    locs = [num for loc in g_feature for num in loc]

    pe = torch.zeros(len(locs), d_model)
    position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.view(n_objects * n_objects, rel_cfg.d_g)

def collate_fn(batch):
    """
    Data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    default_collate = torch.utils.data.dataloader.default_collate
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey == 'mask':
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        elif key == 'depth':
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])

    interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
    collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    return collated_batch

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem
def main_worker(gpu, args, extractor, total3d_model, vsd_3d_encoder):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    dataset = VSD_3D_Custom(img_path='./custom',json_path='./custom')
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,num_workers=1,collate_fn=dataset.collate_fn)
    trainer = Trainer(args, extractor, total3d_model, vsd_3d_encoder, dataloader)
    trainer.test()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()

    args.predict_custom = True # 如果是开放世界测试，需要引入total3d和fastercnn
    args.VL_pretrain = False # 控制对视觉语言模型进行预训练
    if args.predict_custom:
        # build faster rcnn
        print('start to build faster rcnn model')
        extractor = build_model()
        print('faster rcnn model build finish')
        # build total3d model
        print('start to build total3d model')
        total3d_args = total3d_parse_args()
        total3d_cfg = CONFIG(os.path.join('Total3DUnderstanding',total3d_args.config))
        total3d_cfg.update_config(total3d_args.__dict__)
        initiate_environment(total3d_cfg.config)
        '''Load net'''
        checkpoint = CheckpointIO(total3d_cfg)
        cfg = mount_external_config(total3d_cfg)
        # device = load_device(total3d_cfg)
        total3d_model = load_model(total3d_cfg, device='cpu')
        checkpoint.register_modules(net=total3d_model)
        '''Load existing checkpoint'''
        checkpoint.parse_checkpoint()
        print('total3d model build finish')
    else:
        total3d_model = None
        extractor = None

    # build 3dvsd encoder
    print('start to build vsd 3d encoder model')
    vsd_3d_encoder = Model()
    print('vsd 3d encoder model build finish')
    ##############################################
    args.num_workers = 4
    args.backbone = 'VL-T5/t5-base'
    args.output = 'VL-T5/snap/VSD_3D/final/vsd2/VLT5'
    args.num_beams = 5
    args.batch_size = 1
    args.max_text_length = 40
    args.save_result_path = 'custom_save'
    args.visualize = True
    ##############################################
    main_worker(0, args, extractor, total3d_model, vsd_3d_encoder)

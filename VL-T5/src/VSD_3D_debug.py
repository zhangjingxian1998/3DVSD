# 正式训练时的输入模板变化, 还要额外训练一个3DVSD的encoder, 模型需要学习的提示词增加一个<TGT>
# 与之对比的输出是描述性语句
import sys
import os
current_path = ('/').join((sys.path[0].split('/'))[:-3])
sys.path.append(os.path.join(current_path, '3DVSD'))
sys.path.append(os.path.join(current_path, '3DVSD/Total3DUnderstanding'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from packaging import version

from tqdm import tqdm
import torch
import logging
import numpy as np

from param import parse_args

from VSD_3D_data import get_loader
from utils import LossMeter, set_global_logging_level
import dist_utils
# import wandb
from Total3DUnderstanding.net_utils.utils import load_model, CheckpointIO
from Total3DUnderstanding.utils.param import parse_args as total3d_parse_args
from Total3DUnderstanding.configs.config_utils import CONFIG
from vsd_3d.models import Model
import time
# set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False
predict_list = ["to the left of", "to the right of", "in front of","next to", "above","behind","under","on","in"]
# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, total3d_model, vsd_3d_encoder, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

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

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')


    def train(self):
        device = next(self.model.parameters()).device
        self.vsd_3d_encoder = self.vsd_3d_encoder.to(device)
        if self.args.vsd_pretrain:
            for param in self.model.named_parameters():
                param[1].requires_grad = False
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,

            }

            quesid2ans = {}
            time_start = time.time()
            # split_word = '<extra_id_99>'
            # split_id = self.tokenizer.encode(split_word, return_tensors="pt", add_special_tokens=False)
            for step_i, batch in enumerate(self.train_loader):
                time_0 = time.time()
                r_G, text_prompt, score_loss = self.vsd_3d_encoder(self.args, batch)
                #####################################################################################
                batch['batch_entry']['input_ids'] = self.text_process(batch, text_prompt)
                time_1 = time.time()

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch,r_G)
                        else:
                            results = self.model.train_step(batch,r_G)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch,r_G)
                        time_2 = time.time()
                    else:
                        results = self.model.train_step(batch, r_G)
                        time_2 = time.time()
                        

                loss = results['loss'] + score_loss

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                gradients = [param.grad for param in self.model.parameters()]
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print(f"Gradient of layer {name} is None.")
                for name, param in self.vsd_3d_encoder.named_parameters():
                    if param.grad is None:
                        print(f"Gradient of layer {name} is None.")
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                    desc_str += f' | Loss {loss_meter.val:4f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()
                print('加载数据集耗时:',time_0-time_start)
                print('vsd3d网络处理耗时:',time_1-time_0)
                print('VL网络处理耗时:',time_2-time_1)
                time_start = time.time()
                print('反向传播耗时:',time_start-time_0)
                

            if self.verbose:
                pbar.close()

            # Validation
            score_dict = self.evaluate(self.val_loader)

            if self.verbose:
                valid_score = score_dict['CIDEr'] * 100.
                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''
                log_str += "\nEpoch %d: Valid Raw %0.2f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_epoch, best_valid)

                # wandb_log_dict = {}
                # wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)

                # wandb_log_dict['Valid/score'] = valid_score

                # wandb_log_dict['Valid/raw_score'] = score_dict['overall']
                # for qtype, score in score_dict['perQuestionType'].items():
                #     wandb_log_dict[f'Valid_Qtypes/{qtype}'] = score
                # for atype, score in score_dict['perAnswerType'].items():
                #     if atype == 'yes/no':
                #         atype = 'yes_no'
                #     wandb_log_dict[f'Valid_Atypes/{atype}'] = score

                # wandb.log(wandb_log_dict, step=epoch)
                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

        # Test Set
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)

        target, answer = self.predict(self.test_loader)

        if self.verbose:
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(quesid2ans)

            # evaluator.dump_result(quesid2ans)

            # acc_dict_all = evaluator.evaluate_raw(quesid2ans)
            # acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
            # acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

            # wandb_log_dict = {}
            # wandb_log_dict['Test/overall'] = acc_dict_all['overall']
            # wandb_log_dict['Test/topk_optimal'] = acc_dict_answerable['overall']
            # wandb_log_dict['Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

            # for qtype, score in acc_dict_all['perQuestionType'].items():
            #     wandb_log_dict[f'Test_Qtypes/{qtype}'] = score
            # for atype, score in acc_dict_all['perAnswerType'].items():
            #     if atype == 'yes/no':
            #         atype = 'yes_no'
            #     wandb_log_dict[f'Test_Atypes/{atype}'] = score

            # print(wandb_log_dict)
            # wandb.log(wandb_log_dict)

        # if self.args.submit:
        #     dump_path = os.path.join(self.args.output, 'submit.json')
        #     self.predict(self.submit_test_loader, dump_path)

            # wandb.save(dump_path, base_path=self.args.output)
            # wandb.log({'finished': True})

        if self.args.distributed:
            dist.barrier()
            exit()

    def predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length
            num_beams = self.args.num_beams
            max_length = self.args.gen_max_length
            quesid2ans = {}
            target = []
            answer = []
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction")
            for i, batch in enumerate(loader):
                # if i != 9:
                #     continue
                r_G, text_prompt, score_loss = self.vsd_3d_encoder(self.args, batch)
                batch['batch_entry']['input_ids'] = self.text_process(batch, text_prompt)
                if self.args.distributed:
                    results = self.model.module.test_step(batch, r_G,**gen_kwargs)
                else:
                    results = self.model.test_step(batch, r_G,**gen_kwargs)

                pred_ans = results['pred_ans'][0]
                if self.args.replace_rel:
                    rel_list = batch['batch_entry']['sub_rel_obj']
                    for i,result in enumerate(pred_ans):
                        rel_gt = rel_list[i][1]
                        for rel in predict_list:
                            if rel in result:
                                pred_ans[i] = result.replace(rel,rel_gt)
                                break
                for i,result in enumerate(pred_ans):
                    name = batch['batch_entry']['img_id']
                    # with open(f'/home/zhangjx/All_model/genration_scene/3DVSD/save_img_vsd1/{name}.txt', 'w') as t:
                    #     t.write(result)
                    #     t.close()
                # ques_ids = batch['question_ids']
                ques_ids = batch['batch_entry']['sentences']

                for qid, ans in zip(ques_ids, pred_ans):
                    target.append(qid)
                    answer.append(ans)
                    # quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

        if self.args.distributed:
            dist.barrier()

        return target, answer

    def evaluate(self, loader, dump_path=None):
        # quesid2ans = self.predict(loader, dump_path)
        target, answer = self.predict(loader, dump_path)

        if self.verbose:
            evaluator = loader.evaluator
            # acc_dict = evaluator.evaluate_raw(quesid2ans)
            acc_dict = {}
            topk_score = evaluator.evaluate(target, answer)
            # topk_score = evaluator.evaluate(quesid2ans)
            acc_dict['CIDEr'] = topk_score['CIDEr']

            return acc_dict
    
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
        device = next(self.model.parameters()).device
        self.vsd_3d_encoder = self.vsd_3d_encoder.to(device)
        best_path = os.path.join(self.args.output, 'BEST.pth')
        self.load_checkpoint(best_path)

        target, answer = self.predict(self.test_loader)

        if self.verbose:
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(target, answer)

            test_score = score_dict['CIDEr'] * 100.
            log_str = ''
            log_str += "\nTest %0.2f" % (test_score)
            print(log_str)

def main_worker(gpu, args, total3d_model, vsd_3d_encoder):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12346'
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')


    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, total3d_model, vsd_3d_encoder, train_loader, val_loader, test_loader, train=True)

    if args.submit:
        print(f'Building test submit loader at GPU {gpu}')
        submit_test_loader = get_loader(
            args,
            split='test', mode='val', batch_size=valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
        )
        trainer.submit_test_loader = submit_test_loader
    if args.test_only:
        trainer.test()
    else:
        trainer.train()



if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    # torch.autograd.set_detect_anomaly(True)
    args.world_size = ngpus_per_node

    args.predict_custom = False # 如果是开放世界测试，需要引入total3d和fastercnn
    args.VL_pretrain = False # 控制对视觉语言模型进行预训练
    if args.predict_custom:
        # build faster rcnn

        # build total3d model
        total3d_args = total3d_parse_args()
        total3d_cfg = CONFIG(total3d_args.config)
        total3d_cfg.update_config(total3d_args.__dict__)
        '''Load net'''
        total3d_model = load_model(total3d_cfg)
        checkpoint = CheckpointIO(total3d_cfg)
        checkpoint.register_modules(net=total3d_model)
        '''Load existing checkpoint'''
        checkpoint.parse_checkpoint()
    else:
        total3d_model = None

    # build 3dvsd encoder
    vsd_3d_encoder = Model()

    ##############################################
    # --distributed 
    # --multiGPU
    args.distributed = False
    args.multiGPU = False
    args.train = 'train'
    args.valid = 'val'
    args.test = 'test'
    args.optim = 'adamw'
    args.warmup_ratio = 0.1
    args.clip_grad_norm = 5
    args.lr = 5e-5
    args.epochs = 20
    args.num_workers = 4
    args.backbone = 'VL-T5/t5-base'
    args.load = '/home/zhangjx/All_model/genration_scene/3DVSD/VL-T5/snap/VSD_3D/pretrain/VLT5/BEST'
    # args.backbone = 'VL-T5/bart-base'
    # args.load = 'VL-T5/snap/pretrain/VLBart/Epoch30'
    args.output = '/home/zhangjx/All_model/genration_scene/3DVSD/VL-T5/snap/VSD_3D/final/vsd2/VLT5'
    # args.load = None
    args.num_beams = 5
    args.batch_size = 1
    args.valid_batch_size = 1
    args.local_rank = 1
    args.max_text_length = 40
    args.test_only = True
    args.data = 'VSDv2'
    # args.replace_rel = True
    # args.vsd_pretrain = True
    ##############################################

    main_worker(args.local_rank, args, total3d_model, vsd_3d_encoder)

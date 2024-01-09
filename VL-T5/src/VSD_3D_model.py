import torch
import torch.nn as nn

from modeling_t5 import VLT5
class VLT53DVSD(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, r_G):
        if r_G is None:
            kwargs = {}
        else:
            kwargs = {'r_G':r_G}
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)

        batch = batch['batch_entry']
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True,
                **kwargs
            )
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
            loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, r_G, **kwargs):
        self.eval()
        if r_G is None:
            pass
        else:
            kwargs.update({'r_G':r_G})
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        batch = batch['batch_entry']
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        num_beams = kwargs['num_beams']

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:
            # if self.args.s3_search:
            #     force_words = []
            #     output = self.generate(
            #         input_ids=input_ids,
            #         vis_inputs=(vis_feats, vis_pos),
            #         **kwargs
            #     )
            # else:
            def prefix_allowed_tokens_fn_9(batch_id, sent):
                # 如果是生成的第一个令牌，只允许与前缀匹配的令牌
                if sent[-1] == self.tokenizer.encode('fork',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('on',add_special_tokens=False)]
                    return allow_list
                elif sent[-1] == self.tokenizer.encode('on',add_special_tokens=False)[-1] and sent[-2]==self.tokenizer.encode('fork',add_special_tokens=False)[-1] :
                    allow_list = [self.tokenizer.encode('the',add_special_tokens=False)]
                    return allow_list
                elif sent[-1] == self.tokenizer.encode('the',add_special_tokens=False)[-1] and sent[-2]==self.tokenizer.encode('on',add_special_tokens=False)[-1] :
                    allow_list = [self.tokenizer.encode('plate',add_special_tokens=False)]
                    return allow_list
                else:
                    # 其他情况不限制
                    allow_list = list(range(self.tokenizer.vocab_size))
                    return allow_list
            def prefix_allowed_tokens_fn_11(batch_id, sent):
                allow_list = None
                if sent[-1] == self.tokenizer.encode('asparagus',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('under',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('under',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('asparagus',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('the',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('the',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('under',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('basket',add_special_tokens=False)]

                if sent[-1] == self.tokenizer.encode('squash',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('to',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('to',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('squash',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('the',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('the',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('to',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('left',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('left',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('the',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('and',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('and',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('left',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('down',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('down',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('and',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('of',add_special_tokens=False)]
                elif sent[-1] == self.tokenizer.encode('of',add_special_tokens=False)[-1] and sent[-2] == self.tokenizer.encode('down',add_special_tokens=False)[-1]:
                    allow_list = [self.tokenizer.encode('book',add_special_tokens=False)]
                if allow_list:
                    return allow_list
                pass
            force_word = 'on the plate'

            force_word1 = 'under the basket'
            force_word2 = 'to the left of book'
            force_words_ids = [
                                self.tokenizer(force_word1, add_special_tokens=False).input_ids,
                               self.tokenizer(force_word2, add_special_tokens=False).input_ids,
                               ]
            output = self.generate(
                input_ids=input_ids,
                # force_words_ids=force_words_ids,
                vis_inputs=(vis_feats, vis_pos),
                # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_11,
                # num_return_sequences=num_beams,
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result

from modeling_bart import VLBart
class VLBart3DVSD(VLBart):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, r_G):
        self.eval()
        if r_G is None:
            kwargs = {}
        else:
            kwargs = {'r_G':r_G}
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        batch = batch['batch_entry']
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True,
                **kwargs
            )
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']

            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            # loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, r_G,**kwargs):
        self.eval()
        if r_G is None:
            pass
        else:
            kwargs.update({'r_G':r_G})
        num_beams = kwargs['num_beams']
        max_length = kwargs['max_length']
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        batch = batch['batch_entry']
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result
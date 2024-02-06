import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict, Counter

from src.cot_state import ChainOfThoughtState


class ThorStateTrainer:
    """ This is a default trainer proposed by authors.
    """

    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.scores, self.lines = [], []
        self.re_init()

    def train(self, epoch_from):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            if epoch < epoch_from:
                continue
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            print(result)
            self.re_init()
            score = result['default']
            self.add_instance(result)
            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

    def prepare_step_two(self, aspect_exprs, data):
        context_A_ids, target_ids = [data[w] for w in 'context_A_ids, target_ids'.strip().split(', ')]
        contexts_A = [self.model.tokenizer.decode(ids) for ids in context_A_ids]
        contexts_A = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_A]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_B = []
        for context, target, aspect_expr in zip(contexts_A, targets, aspect_exprs):
            context_B, prompt = ChainOfThoughtState.prompt_for_opinion_inferring(context, target, aspect_expr)
            new_prompts.append(prompt)
            contexts_B.append(context_B)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data
        batch_contexts_B = self.model.tokenizer.batch_encode_plus(contexts_B, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_B = batch_contexts_B.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_B_ids': batch_contexts_B['input_ids'],
            'target_ids': target_ids,
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_three(self, opinion_exprs, data):
        context_B_ids, target_ids = [data[w] for w in 'context_B_ids, target_ids'.strip().split(', ')]
        contexts_B = [self.model.tokenizer.decode(ids) for ids in context_B_ids]
        contexts_B = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_B]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_C = []
        for context, target, opinion_expr in zip(contexts_B, targets, opinion_exprs):
            context_C, prompt = ChainOfThoughtState.prompt_for_emotion_inferring(context, target, opinion_expr)
            new_prompts.append(prompt)
            contexts_C.append(context_C)

        batch_contexts_C = self.model.tokenizer.batch_encode_plus(contexts_C, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_C = batch_contexts_C.data
        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_C_ids': batch_contexts_C['input_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_label(self, polarity_exprs, pre_cxt, data):
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]

        context_C_ids = pre_cxt['context_C_ids']
        contexts_C = [self.model.tokenizer.decode(ids) for ids in context_C_ids]
        contexts_C = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_C]

        new_prompts = []
        for context_C, polarity_expr in zip(contexts_C, polarity_exprs):
            prompt = ChainOfThoughtState.prompt_for_emotion_label(context_C, polarity_expr, self.config.label_list)
            new_prompts.append(prompt)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=len(self.config.label_list))
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length)

        losses = []
        for i, data in enumerate(train_data):
            step_one_inferred_output = self.model.generate(**data)

            step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
            step_two_inferred_output = self.model.generate(**step_one_inferred_data)

            step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
            step_three_inferred_output = self.model.generate(**step_two_inferred_data)

            step_label_data = self.prepare_step_label(step_three_inferred_output, step_two_inferred_data, data)
            loss = self.model(**step_label_data)
            losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)

            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def do_infer_iter(self, dataLoader=None):
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                step_one_inferred_output = self.model.generate(**data)

                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
                step_three_inferred_output = self.model.generate(**step_two_inferred_data)

                step_label_data = self.prepare_step_label(step_three_inferred_output, step_two_inferred_data, data)
                output = self.model.evaluate(**step_label_data)
                yield data, output

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        for data, output in self.do_infer_iter(dataLoader):
            self.add_output(data, output)
        result = self.report_score(mode=mode)
        return result

    def final_infer(self, dataLoader):
        self.model.eval()
        result = defaultdict(list)
        for _, output in self.do_infer_iter(dataLoader):
            result["total"] += output
        return result

    def load_from_epoch(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])

    def load_from_path(self, state_path=None):
        self.model.load_state_dict(torch.load(state_path, map_location=self.config.device)['model'])

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total']

    def add_output(self, data, output):
        gold = data['input_labels']
        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()

    def report_score(self, mode='valid'):
        c = Counter()
        for l in self.preds['total']:
            c[l] += 1

        res = {}
        res['Acc'] = accuracy_score(self.golds['total'], self.preds['total'])
        res["F1"] = f1_score(self.golds['total'], self.preds['total'], average='macro', labels=list(range(len(self.config.label_list))))
        res['default'] = res['F1']
        res['mode'] = mode
        res['labels'] = c
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res

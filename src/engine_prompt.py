import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict, Counter


class PromptTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.scores, self.lines = [], []
        self.re_init()

    def train(self, epoch_from=0):
        best_score, best_iter = 0, -1
        epoch = -1
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

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        losses = []
        for i, data in enumerate(train_data):
            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                self.add_output(data, output)
        result = self.report_score(mode=mode)
        return result

    def infer_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        result = defaultdict(list)
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                result["total"] += output
        return result

    def load_from_epoch(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])

    def load_from_path(self, state_path=None):
        self.model.load_state_dict(torch.load(state_path, map_location=self.config.device)['model'])

    def final_infer(self, dataLoader):
        self.model.eval()
        res = self.infer_step(self.valid_loader if dataLoader is None else dataLoader)
        self.add_instance(res)
        return res

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
        labels = list(range(len(self.config.label_list)))

        c = Counter()
        for l in self.preds['total']:
            c[l] += 1

        res = {}
        res['Acc'] = accuracy_score(self.golds['total'], self.preds['total'])
        res["F1"] = f1_score(self.golds['total'], self.preds['total'], average='macro', labels=labels)
        res['default'] = res['F1']
        res['mode'] = mode
        res['labels'] = c
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res


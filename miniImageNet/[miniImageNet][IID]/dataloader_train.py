"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random
import pickle as pkl
from functions import *
import torch
import torch.nn.functional as F

class miniImageNetGenerator(object):

    def __init__(self, data_path, args, max_iter=None):
        super(miniImageNetGenerator, self).__init__()
        self.data_path = data_path
        self.max_iter = max_iter
        self.num_iter = 0
        self.data_dict = self._load_data(self.data_path)
        self.args = args
        self.num_user = args.total_user
        self.user_data = torch.zeros(self.num_user, int(len(self.data_dict[next(iter(self.data_dict))])), 3, 84, 84)
        self.user_label = torch.zeros(self.num_user, int(len(self.data_dict[next(iter(self.data_dict))]))).long()
        self.Initialize()

    def _load_data(self, data_file):
        dataset = self.load_data(data_file)
        data = dataset['data']
        labels = dataset['labels']
        label2ind = self.buildLabelIndex(labels)

        return {key: torch.tensor(data[val]).permute([0,3,1,2]) for (key, val) in label2ind.items()}

    def load_data(self, data_file):
        try:
            with open(data_file, 'rb') as fo:
                data = pkl.load(fo)
            return data
        except:
            with open(data_file, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data

    def buildLabelIndex(self, labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            x_spt, y_spt, x_qry, y_qry = self.smallSample()

            return (self.num_iter - 1), x_spt, y_spt, x_qry, y_qry
        else:
            raise StopIteration()


    def augment(self, img):

        # random cropping
        npad = (8, 8, 8, 8)
        img = F.pad(img, npad) # [3,84,84]
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[:,y:y + 84, x:x + 84]

        return img

    def Initialize(self):
        visit_user = {i: 0 for i in range(self.num_user)}
        key_list = sorted(self.data_dict.keys())
        assert len(key_list) % self.num_user == 0, "data len isn't multiplication of num of users."
        for cls in key_list:
            _imgs = self.data_dict[cls]
            spSize = int(len(_imgs) / self.args.n_split)  # 300
            for spIdx in range(self.args.n_split): # 0,1
                split = _imgs[spIdx*spSize:(spIdx+1)*spSize]
                user = random.sample(list(visit_user),1)[0]
                userSpIdx = visit_user[user]
                self.user_data[user,userSpIdx*spSize:(userSpIdx+1)*spSize] = split
                self.user_label[user,userSpIdx*spSize:(userSpIdx+1)*spSize] = cls
                visit_user[user] += 1
                if visit_user[user] == self.args.n_split:
                    del visit_user[user]

    def smallSample(self):
        def DivideSet(container, size):
            out = random.sample(container, size)
            container = container - set(out)
            return out, list(container)

        n_user = self.args.n_user
        size = len(self.user_data[0])  # 600
        dSize = int(size * self.args.data_ratio)  # 240
        x_spt = torch.zeros(n_user, int(dSize / 2), 3, 84, 84)  # [10,150,3,84,84]
        x_qry = torch.zeros(n_user, int(dSize / 2), 3, 84, 84)
        y_spt = torch.zeros(n_user, int(dSize / 2), dtype=torch.long)
        y_qry = torch.zeros(n_user, int(dSize / 2), dtype=torch.long)
        users = random.sample(range(len(self.user_data)), n_user)
        for Oidx, idx in enumerate(users):  # 0~9
            bSize = int(size / self.args.n_split)  # 300
            sSize = int(bSize * self.args.data_ratio / 2)  # 60
            for s in range(self.args.n_split):
                allIdx = set(random.sample(range(s * bSize, (s + 1) * bSize), int(bSize * self.args.data_ratio)))
                sptIdx, qryIdx = DivideSet(allIdx, sSize)
                sStart, sEnd = s * sSize, (s + 1) * sSize
                x_spt[Oidx, sStart:sEnd] = torch.stack([self.augment(i) for i in self.user_data[idx, sptIdx]])
                x_qry[Oidx, sStart:sEnd] = torch.stack([self.augment(i) for i in self.user_data[idx, qryIdx]])
                y_spt[Oidx, sStart:sEnd] = self.user_label[idx, sptIdx]
                y_qry[Oidx, sStart:sEnd] = self.user_label[idx, qryIdx]

        return x_spt, y_spt, x_qry, y_qry


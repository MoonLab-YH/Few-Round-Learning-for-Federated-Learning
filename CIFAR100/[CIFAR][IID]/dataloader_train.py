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

class CIFARGenerator(object):

    def __init__(self, train_path, test_path, args, max_iter=None):
        super(CIFARGenerator, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.max_iter = max_iter
        self.num_iter = 0
        self.train_dict = self.unpickle(self.train_path)
        self.test_dict = self.unpickle(self.test_path)
        self.data_dict = self.MergeDict()
        self.args = args
        self.num_user = args.total_user
        self.user_data = torch.zeros(self.num_user, 600, 3, 32, 32)
        self.user_label = torch.zeros(self.num_user, 600).long()
        self.Initialize()

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'fine_labels']
        label2ind = self.buildLabelIndex(labels)

        return {key: torch.tensor(data[val]).view([len(val), 3, 32, 32]) for (key, val) in label2ind.items()}

    def MergeDict(self):
        Dict = {key:None for key in range(64)}
        for key in Dict.keys():
            Dict[key] = torch.cat((self.train_dict[key], self.test_dict[key]), dim=0)

        return Dict

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
        npad = (3, 3, 3, 3)
        img = F.pad(img, npad)
        x = random.randint(0, 6)
        y = random.randint(0, 6)
        img = img[:,y:y + 32, x:x + 32]

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
        x_spt = torch.zeros(n_user, int(dSize / 2), 3, 32, 32)  # [10,150,3,32,32]
        x_qry = torch.zeros(n_user, int(dSize / 2), 3, 32, 32)
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


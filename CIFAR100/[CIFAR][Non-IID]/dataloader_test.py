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

    def __init__(self,  train_path, test_path, args, max_iter=None):
        self.train_path = train_path
        self.test_path = test_path
        self.max_iter = max_iter
        self.num_iter = 0
        self.train_dict = self.unpickle(self.train_path)
        self.test_dict = self.unpickle(self.test_path)
        self.data_dict = self.MergeDict()
        self.args = args
        self.num_user = args.test_total_user
        self.user_data = torch.zeros(self.num_user, 300, 3, 32, 32)
        self.user_label = torch.zeros(self.num_user, 300)

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'fine_labels']
        label2ind = self.buildLabelIndex(labels)

        return {key: torch.tensor(data[val]).view([len(val), 3, 32, 32]) for (key, val) in label2ind.items()}

    def MergeDict(self):
        Dict = {key:None for key in range(80,100)}
        for key in Dict.keys():
            Dict[key] = torch.cat((self.train_dict[key], self.test_dict[key]), dim=0)

        return Dict

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
            x_spt, y_spt, x_qry, y_qry = self.sample()

            return (self.num_iter - 1), x_spt, y_spt, x_qry, y_qry
        else:
            raise StopIteration()

    def Initialize(self):
        visit_user = {i: 0 for i in range(self.num_user)}
        key_list = self.data_dict.keys()
        small_key_list = random.sample(key_list, self.args.n_cls_at_test)
        for cls in small_key_list:
            _imgs = self.data_dict[cls]
            spSize = int(len(_imgs)/4)  # 150
            for spIdx in range(4):  # 0,1
                split = _imgs[spIdx * spSize:(spIdx + 1) * spSize]
                user = random.sample(list(visit_user), 1)[0]
                userSpIdx = visit_user[user]
                self.user_data[user, userSpIdx * spSize:(userSpIdx + 1) * spSize] = split
                self.user_label[user, userSpIdx * spSize:(userSpIdx + 1) * spSize] = cls
                visit_user[user] += 1
                if visit_user[user] == self.args.n_split:
                    del visit_user[user]

    def sample(self):
        def DivideSet(container, size):
            out = random.sample(container,size)
            container = container - set(out)
            return out, list(container)
        self.Initialize()

        n_user = self.args.n_user
        size = len(self.user_data[0]) # 300
        x_spt = torch.zeros(n_user, int(size / 2), 3, 32, 32) # [10,150,3,32,32]
        x_qry = torch.zeros(n_user, int(size / 2), 3, 32, 32)
        y_spt = torch.zeros(n_user, int(size / 2), dtype=int)
        y_qry = torch.zeros(n_user, int(size / 2), dtype=int)
        for idx in range(n_user): # 0~9
            bSize = int(size / self.args.n_split) # 150
            sSize = int(bSize / 2) # 75
            for s in range(self.args.n_split):
                allIdx = set(range(s*bSize,(s+1)*bSize))
                sptIdx, qryIdx = DivideSet(allIdx, sSize)
                sStart, sEnd = s * sSize, (s+1) * sSize
                x_spt[idx, sStart:sEnd] = self.user_data[idx, sptIdx]
                x_qry[idx, sStart:sEnd] = self.user_data[idx, qryIdx]
                y_spt[idx, sStart:sEnd] = self.user_label[idx, sptIdx]
                y_qry[idx, sStart:sEnd] = self.user_label[idx, qryIdx]

        return x_spt, y_spt, x_qry.view(-1, *x_qry.shape[2:]), y_qry.view(-1)


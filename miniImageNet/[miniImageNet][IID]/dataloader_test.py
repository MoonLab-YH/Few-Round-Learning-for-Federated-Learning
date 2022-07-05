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
        self.num_user = args.test_total_user

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
            x_spt, y_spt, x_qry, y_qry = self.sample()

            return (self.num_iter - 1), x_spt, y_spt, x_qry, y_qry
        else:
            raise StopIteration()


    def sample(self):
        n_user = self.args.n_user # 10
        ncls = self.args.n_cls_at_test # 5
        nDataCls = 600
        nDataUserCls = int(nDataCls/ n_user / 2) # 30
        x_spt = torch.zeros(n_user, nDataUserCls * ncls, 3, 84, 84) # [10,150,3,84,84]
        x_qry = torch.zeros(n_user, nDataUserCls * ncls, 3, 84, 84)
        y_spt = torch.zeros(n_user, nDataUserCls * ncls, dtype=int)
        y_qry = torch.zeros(n_user, nDataUserCls * ncls, dtype=int)
        key_list = self.data_dict.keys()
        small_key_list = random.sample(key_list, ncls)
        bigData = {cls:self.data_dict[cls] for cls in small_key_list}
        for cls in small_key_list: # Shuffle data.
            bigData[cls] = bigData[cls][np.random.permutation(nDataCls)]
            bigData[cls] = bigData[cls].view(n_user, 2, -1, 3, 84, 84) # [10,2,30,3,84,84]
        for user in range(n_user):
            for idx, cls in enumerate(small_key_list):
                x_spt[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = bigData[cls][user][0]
                x_qry[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = bigData[cls][user][1]
                y_spt[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = cls
                y_qry[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = cls

        return x_spt, y_spt, x_qry.view(-1, *x_qry.shape[2:]), y_qry.view(-1)


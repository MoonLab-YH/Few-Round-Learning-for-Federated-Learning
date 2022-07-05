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


    def sample(self):
        n_user = self.args.n_user # 10
        ncls = self.args.n_cls_at_test # 5
        nDataCls = 600
        nDataUserCls = int(nDataCls/ n_user / 2) # 30
        x_spt = torch.zeros(n_user, nDataUserCls * ncls, 3, 32, 32) # [10,150,3,84,84]
        x_qry = torch.zeros(n_user, nDataUserCls * ncls, 3, 32, 32)
        y_spt = torch.zeros(n_user, nDataUserCls * ncls, dtype=int)
        y_qry = torch.zeros(n_user, nDataUserCls * ncls, dtype=int)
        key_list = self.data_dict.keys()
        small_key_list = random.sample(key_list, ncls)
        bigData = {cls:self.data_dict[cls] for cls in small_key_list}
        for cls in small_key_list: # Shuffle data.
            bigData[cls] = bigData[cls][np.random.permutation(nDataCls)]
            bigData[cls] = bigData[cls].view(n_user, 2, -1, 3, 32, 32) # [10,2,30,3,84,84]
        for user in range(n_user):
            for idx, cls in enumerate(small_key_list):
                x_spt[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = bigData[cls][user][0]
                x_qry[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = bigData[cls][user][1]
                y_spt[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = cls
                y_qry[user][idx*nDataUserCls:(idx+1)*nDataUserCls] = cls

        return x_spt, y_spt, x_qry.view(-1, *x_qry.shape[2:]), y_qry.view(-1)


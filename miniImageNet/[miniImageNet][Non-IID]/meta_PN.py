import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import copy

from    learner import Learner
from    copy import deepcopy
from functions import *

class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.n_spt = args.n_spt
        self.n_qry = args.n_qry
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.args = args
        self.net = Learner(config, args.imgc, args.imgsz)

    def forward(self, step, x_spt, y_spt, x_qry, y_qry, device):

        num_user, setsz, c_, h, w = x_spt.size()
        classes = torch.unique(y_spt)
        global_net = copy.deepcopy(self.net)
        Post_prototypes = {i.item(): torch.ones((64), dtype=torch.float).to(device) for i in classes}
        Local_prototypes = {i: None for i in range(num_user)}

        for round in range(1, self.args.round + 1): # update theta
            weights = []
            Pre_prototypes = {i.item(): [0, torch.ones((self.args.n_split, 64), dtype=torch.float).to(device)] for i in classes}
            for i in range(num_user):
                _x_spt = x_spt[i]; _y_spt = y_spt[i]
                local_net = copy.deepcopy(global_net)
                local_optim = optim.SGD(local_net.parameters(), lr=self.update_lr, weight_decay=5e-4)
                for ep in range(self.args.local_ep):
                    for n_batch in range(int(len(_x_spt)/self.args.batch_size)):
                        _x_spt_B, _y_spt_B = MakeBatch(_x_spt, _y_spt, n_batch, self.args, train=True)
                        L_y_spt_B = LocalLabel(_y_spt_B)
                        sup_feat = self.net(_x_spt_B, vars=local_net.parameters(), bn_training=True).squeeze()
                        prototype = self.make_prototype(sup_feat, _y_spt_B, device)
                        prob = PN_pred(prototype, sup_feat)  # [25,5]
                        Lloss = F.cross_entropy(prob, L_y_spt_B)
                        Gloss = self.GlobalLoss(Post_prototypes, sup_feat, _y_spt_B) if round != 1 else 0
                        loss = Lloss + 0.2 * Gloss
                        local_net.zero_grad()
                        loss.backward()
                        local_optim.step()
                full_feat = self.net(_x_spt, vars=local_net.parameters(), bn_training=True).squeeze()
                full_proto = self.make_prototype(full_feat, _y_spt, device).detach()
                Local_prototypes[i] = full_proto
                weights.append(local_net.parameters())
                UploadProto(Pre_prototypes, full_proto, _y_spt)
            with torch.no_grad():
                avg_weight = average_weights(weights)
                average_prototypes(Pre_prototypes, Post_prototypes)
                for Gparam, Aparam in zip(global_net.parameters(), avg_weight):
                    Gparam.copy_(Aparam)

        meta_weights, meta_acc = [], []
        for i in range(num_user): # meta-update phi
            _x_qry, _y_qry = x_qry[i], y_qry[i]
            local_net = copy.deepcopy(self.net) # Phi
            local_optim = optim.SGD(local_net.parameters(), lr=self.meta_lr, weight_decay=5e-4)
            for ep in range(self.args.local_ep):
                for n_batch in range(int(len(_x_qry) / self.args.batch_size)):
                    _x_qry_B, _y_qry_B = MakeBatch(_x_qry, _y_qry, n_batch, self.args, train=True)
                    L_y_qry_B = LocalLabel(_y_qry_B)
                    if n_batch == 0:
                        qry_feat = self.net(_x_qry_B, vars=global_net.parameters(), bn_training=True).squeeze()
                        meta_pred = PN_pred(Local_prototypes[i], qry_feat)
                        Lloss = F.cross_entropy(meta_pred, L_y_qry_B)
                        Gloss = self.GlobalLoss(Post_prototypes, qry_feat, _y_qry_B)
                        meta_loss = Lloss + 0.2 * Gloss
                        meta_grad = torch.autograd.grad(meta_loss, global_net.parameters())
                        with torch.no_grad():
                            for Mparam, Mgrad in zip(local_net.parameters(), meta_grad):
                                Mparam.grad = Mgrad
                        local_optim.step()
                    else:
                        qry_feat = self.net(_x_qry_B, vars=local_net.parameters(), bn_training=True).squeeze()
                        meta_pred = PN_pred(Local_prototypes[i], qry_feat)
                        Lloss = F.cross_entropy(meta_pred, L_y_qry_B)
                        Gloss = self.GlobalLoss(Post_prototypes, qry_feat, _y_qry_B)
                        meta_loss = Lloss + 0.2 * Gloss
                        if n_batch+1 == int(len(_x_qry) / self.args.batch_size):
                            meta_acc.append((meta_pred.argmax(dim=1)==L_y_qry_B).float().mean())
                        local_net.zero_grad()
                        meta_loss.backward()
                        local_optim.step()
            meta_weights.append(local_net.parameters())
        with torch.no_grad():
            avg_weight = average_weights(meta_weights)
            for Gparam, Aparam in zip(self.net.parameters(), avg_weight):
                Gparam.copy_(Aparam)

        return torch.tensor(meta_acc).mean().item()

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, device):
        # we finetunning on the copied model instead of self.net
        net = copy.deepcopy(self.net)
        num_user, setsz, c_, h, w = x_spt.size()
        classes = torch.unique(y_spt)
        Post_prototypes = {i.item(): torch.ones((64), dtype=torch.float).to(device) for i in classes}

        for round in range(1, self.args.round + 1):
            weights = []
            Pre_prototypes = {i.item(): [0, torch.ones((self.args.test_n_split, 64), dtype=torch.float).to(device)] for i in classes}
            for i in range(num_user):
                _x_spt = x_spt[i]; _y_spt = y_spt[i]
                local_net = deepcopy(net)
                optimizer = torch.optim.SGD(local_net.parameters(), lr=self.update_lr, weight_decay=5e-4)
                for ep in range(self.args.local_ep):
                    for n_batch in range(int(len(_x_spt) / self.args.test_batch_size)):
                        _x_spt_B, _y_spt_B = MakeBatch(_x_spt, _y_spt, n_batch, self.args, train=False)
                        L_y_spt_B = LocalLabel(_y_spt_B)
                        sup_feat = net(_x_spt_B, vars=local_net.parameters(), bn_training=True).squeeze()  # [25,32]
                        prototype = self.make_prototype(sup_feat, _y_spt_B, device)
                        prob = PN_pred(prototype, sup_feat)  # [2,32]
                        Lloss = F.cross_entropy(prob, L_y_spt_B)
                        Gloss = self.GlobalLoss(Post_prototypes, sup_feat, _y_spt_B) if round != 1 else 0
                        loss = Lloss + 0.2 * Gloss
                        local_net.zero_grad()
                        loss.backward()
                        optimizer.step()
                full_feat = self.net(_x_spt, vars=local_net.parameters(), bn_training=True).squeeze()
                full_proto = self.make_prototype(full_feat, _y_spt, device).detach()
                weights.append(list(local_net.parameters()))
                UploadProto(Pre_prototypes, full_proto, _y_spt)
            with torch.no_grad():
                w_glob = average_weights(weights)
                average_prototypes(Pre_prototypes, Post_prototypes)
                for Gparam, Aparam in zip(net.parameters(), w_glob):
                    Gparam.copy_(Aparam)

        with torch.no_grad():
            qry_feat = net(x_qry, vars=net.parameters(), bn_training=True).squeeze()  # [100,32]
            acc = Qry_PN_Acc(Post_prototypes, qry_feat, y_qry).item()  # [25,5]

        del net
        del local_net
        del qry_feat
        return acc


    def make_prototype(self, feats, y_spt, device):
        labels = sorted(torch.unique(y_spt))
        C = feats.size(1)
        prototypes = torch.ones((len(labels), C), dtype=torch.float).to(device)
        for lIdx, label in enumerate(labels):
            pos = (y_spt == label)
            feat = feats[pos]
            prototype = feat.mean(dim=0)
            prototypes[lIdx] = prototype
        return prototypes

    def GlobalLoss(self, Post_protos, sup_feat, _y_spt):
        key_list = sorted(Post_protos.keys())
        global_proto = torch.zeros(len(key_list),64).to(sup_feat.device) # [N,32]
        for idx, key in enumerate(key_list):
            global_proto[idx] = Post_protos[key]
        pred = PN_pred(global_proto, sup_feat)
        gLabel = GlobalLabel(Post_protos, _y_spt) # [300]
        loss = F.cross_entropy(pred,gLabel)
        return loss

    # def Global_DC_Loss(self, Post_protos, sup_feat, _y_spt):
    #     key_list = sorted(Post_protos.keys())
    #     global_proto = torch.zeros(len(key_list),64).to(sup_feat.device) # [N,32]
    #     for idx, key in enumerate(key_list):
    #         global_proto[idx] = Post_protos[key]
    #     input1 = global_proto.unsqueeze(dim=0)[(...,) + (None,) * 2] # [1,Nc,32,1,1]
    #     input2 = sup_feat.unsqueeze(dim=1) # [300,1,32,6,6]
    #     dist = -(input2-input1).pow(2).sum(dim=2) # [300,Nc,6,6]
    #     label = torch.ones_like(dist[:,0,:,:], dtype=int) # [300,6,6]
    #     gLabel = GlobalLabel(Post_protos, _y_spt) # [300]
    #     for i in range(len(gLabel)):
    #         label[i] = gLabel[i] # [300,6,6]
    #     pred = dist.permute([0,2,3,1]).contiguous().view(-1,len(key_list)) # [10800,Nc]
    #     label = label.flatten() # [10800]
    #
    #     loss = F.cross_entropy(pred,label)
    #     return loss

def PN_pred(prototype, qry_feat):
    # qryfeat = [25,32], prototype = [5,32]
    distance = qry_feat.unsqueeze(dim=1) - prototype # [25,5,32]
    distance = distance.pow(2).sum(dim=2)
    return -distance # [25,5]

def Meta_PN_Loss(Post_protos, qry_feat, _y_qry):
    key_list = sorted(Post_protos.keys())
    global_proto = torch.zeros(len(key_list), 64).to(qry_feat.device)  # [N,32]
    for idx, key in enumerate(key_list):
        global_proto[idx] = Post_protos[key]
    distance = qry_feat.unsqueeze(dim=1) - global_proto  # [300,Nc,32]
    pred = -distance.pow(2).sum(dim=2)
    gLabel = GlobalLabel(Post_protos, _y_qry)  # [300]

    loss = F.cross_entropy(pred, gLabel)
    pred_int = pred.argmax(dim=1)
    acc = (pred_int == gLabel).float().mean()
    return loss, acc

def Qry_PN_Acc(Post_protos, qry_feat, y_qry):
    key_list = sorted(Post_protos.keys())
    global_proto = torch.zeros(len(key_list), 64).to(qry_feat.device)  # [Nc,32]
    for idx, key in enumerate(key_list):
        global_proto[idx] = Post_protos[key]
    pred = PN_pred(global_proto, qry_feat).argmax(dim=1)
    gLabel = GlobalLabel(Post_protos, y_qry)  # [300]
    acc = (pred == gLabel).float().mean()

    return acc

def average_weights(weights):
    out = weights[0]
    for idx in range(1, len(weights)):
        for widx in range(len(out)):
            out[widx] = nn.Parameter(out[widx] + weights[idx][widx])

    for widx in range(len(out)):
        out[widx] = nn.Parameter(out[widx] / len(weights))

    return out

def average_prototypes(Pre_protos, Post_protos):
    for cls in Pre_protos.keys():
        n_splits = Pre_protos[cls][0]
        protos = Pre_protos[cls][1][:n_splits].mean(dim=0)
        Post_protos[cls] = protos

def LocalLabel(_y_spt):
    out = torch.zeros_like(_y_spt, dtype=torch.long)
    for idx, label in enumerate(_y_spt.unique()):
        pos = (_y_spt == label)
        out[pos] = idx

    return out

def GlobalLabel(Post_protos, _y_spt):
    out = torch.zeros_like(_y_spt).long()
    key_list = sorted(list(Post_protos))
    for label in _y_spt:
        gLabel = key_list.index(label)
        pos = (_y_spt == label)
        out[pos] = gLabel

    return out

def UploadProto(prototypes, prototype, _y_spt):
    for idx, label in enumerate(_y_spt.unique()):
        nth_split = prototypes[label.item()][0]
        prototypes[label.item()][1][nth_split] = prototype[idx]
        prototypes[label.item()][0] += 1

def MakeBatch(_x_spt, _y_spt, n_batch, args, train=True):
    batch_size = args.batch_size if train else args.test_batch_size # 30 or 10
    x_out = torch.zeros(batch_size, *_x_spt.shape[1:]).to(_x_spt.device)
    y_out = torch.zeros(batch_size, dtype=torch.long).to(_y_spt.device)
    sSize = int(batch_size / args.n_split) 
    bSize = int(len(_x_spt) / args.n_split) 
    for split in range(args.n_split):
        sStart, bStart = split*sSize, split*bSize + n_batch*sSize
        sEnd, bEnd = (split+1)*sSize, split*bSize + (n_batch+1)*sSize
        x_out[sStart:sEnd] = _x_spt[bStart:bEnd]
        y_out[sStart:sEnd] = _y_spt[bStart:bEnd]

    return x_out, y_out

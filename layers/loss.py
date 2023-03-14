import torch
import torch.nn as nn
import random

class CalPairDiff(nn.Module):
    def __init__(self, type):
        super(CalPairDiff, self).__init__()
        if type.lower() == 'l1':
            self.cal_loss = nn.L1Loss()
        elif type.lower() == 'l2':
            self.cal_loss = nn.MSELoss()
        elif type.lower() == 'kldiv':
            self.cal_loss = nn.KLDivLoss()
        
    def forward(self, predicts, target):
        B = predicts.shape[0]
        ori_pre = predicts
        ori_tar = target
        index = list(range(B))
        random.shuffle(index)
        predicts = predicts[index]
        target = target[index]
        v1 = ori_pre - predicts
        v2 = ori_tar - target
        loss = self.cal_loss(v1, v2)
        return loss

class DiffLoss(nn.Module):
    def __init__(self, type):
        super(DiffLoss, self).__init__()
        self.crit = CalPairDiff(type)

    def forward(self, predicts, target):
        loss = self.crit(predicts, target)
        return loss

if __name__ == '__main__':
    loss = DiffLoss()
    a1=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=float)
    a2=torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=float)
    l1 = loss(a1, a2)

    b1=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    b2=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    l2 = loss(b1, b2)

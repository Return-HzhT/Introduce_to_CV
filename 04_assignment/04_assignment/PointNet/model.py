from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet
# ----------TODO------------


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, d=1024):
        super(PointNetfeat, self).__init__()
        self.d = d
        self.global_feat = global_feat
        self.linear1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, self.d)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.pool = nn.MaxPool1d(kernel_size=1024)

    def forward(self, x):
        x = self.relu1(self.bn1(self.linear1(x)))
        local_feature = x
        x = self.relu2(self.bn2(self.linear2(x)))
        x = self.bn3(self.linear3(x))
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        if self.global_feat:
            x = x.view(-1, self.d)
            return x
        else:
            x = x.repeat(1, 1024, 1)
            return torch.cat([local_feature, x], dim=2)


class PointNetCls1024D(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=True, d=1024)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                nn.Linear(512, 256), nn.ReLU(),
                                nn.Linear(256, self.k))

    def forward(self, x):
        # for visulization
        x = self.feat.relu1(self.feat.bn1(self.feat.linear1(x)))
        x = self.feat.relu2(self.feat.bn2(self.feat.linear2(x)))
        x = self.feat.bn3(self.feat.linear3(x))
        vis_feature = x
        x = x.transpose(1, 2)
        x = self.feat.pool(x)
        x = x.transpose(1, 2)
        x = x.view(-1, self.feat.d)
        # x = self.feat(x)
        x = self.fc(x)
        return F.log_softmax(
            x, dim=1
        ), vis_feature  # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=True, d=256)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                nn.Linear(128, self.k))

    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        vis_feature = None
        return F.log_softmax(x, dim=1), vis_feature


class PointNetSeg(nn.Module):
    def __init__(self, k=2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False, d=1024)
        self.fc = nn.Sequential(nn.Linear(1088, 512),
                                nn.BatchNorm1d(num_features=1024), nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(num_features=1024), nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.BatchNorm1d(num_features=1024), nn.ReLU(),
                                nn.Linear(128, self.k))

    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)

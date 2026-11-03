"""Two standard deep baselines for skeleton exercise assessment.

LSTM   : 2-layer LSTM (128) over flattened joints per frame, mean+last pooling.
STGCN  : spatio-temporal graph convolution (Yan et al. 2018 style), 4 blocks (64,64,128,128),
         graph = k-nearest-neighbour joints in the mean pose (so one code works for
         Kinect / MediaPipe / Vicon marker sets) plus a learnable residual adjacency
         (as in adaptive GCNs).
Both output one value: logit (classification) or score (regression).
"""
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, J, hidden=128):
        super().__init__()
        self.inp = nn.Linear(J * 3, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 1))

    def forward(self, x):                      # x (B, T, J, 3)
        h, _ = self.rnn(torch.relu(self.inp(x.flatten(2))))
        return self.head(torch.cat([h.mean(1), h[:, -1]], 1)).squeeze(1)


def knn_adjacency(mean_pose, k=3):
    J = len(mean_pose)
    d = np.linalg.norm(mean_pose[:, None] - mean_pose[None], axis=-1)
    A = np.eye(J)
    for i in range(J):
        for j in np.argsort(d[i])[1:k + 1]:
            A[i, j] = A[j, i] = 1
    D = np.diag(1 / np.sqrt(A.sum(1)))
    return (D @ A @ D).astype(np.float32)


class Block(nn.Module):
    def __init__(self, cin, cout, A, stride=1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.B = nn.Parameter(torch.zeros_like(self.A))
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(0.2))
        self.res = nn.Identity() if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, (stride, 1))

    def forward(self, x):                      # x (B, C, T, J)
        y = torch.einsum("bctj,jk->bctk", self.gcn(x), self.A + self.B)
        return torch.relu(self.tcn(y) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, J, A):
        super().__init__()
        self.bn = nn.BatchNorm1d(J * 3)
        self.blocks = nn.Sequential(Block(3, 64, A), Block(64, 64, A), Block(64, 128, A, 2), Block(128, 128, A))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):                      # x (B, T, J, 3)
        B, T, J, C = x.shape
        x = self.bn(x.flatten(2).transpose(1, 2)).view(B, J, C, T).permute(0, 2, 3, 1)
        return self.head(self.blocks(x).mean((2, 3))).squeeze(1)

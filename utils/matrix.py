import pickle
import itertools
# Basic Library
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML Library
from scipy import stats

# DL Library
import torch

class MIRT_torch(torch.nn.Module):
    def __init__(self, n_p, n_q, rank):
        super().__init__()
        self.rank = rank
        self.n_p, self.n_q = n_p, n_q
        self.p = torch.nn.Embedding(n_p, rank, sparse=True)
        self.q = torch.nn.Embedding(n_q, rank, sparse=True)
    def forward(self, i, j):
        p, q = self.p(i), self.q(j)
        return torch.prod(torch.sigmoid(p+q), dim=1).view(-1,1)

class LN_torch(torch.nn.Module):
    def __init__(self, n_p, n_q, rank):
        super().__init__()
        self.rank = rank
        self.n_p, self.n_q = n_p, n_q
        self.p = torch.nn.Embedding(n_p, rank, sparse=True)
        self.q = torch.nn.Embedding(n_q, rank + 1, sparse=True)
    def forward(self, i, j):
        p, q = self.p(i), self.q(j)
        return torch.sigmoid(torch.sum(p * q.T[:-1].T,axis=1) + q.T[-1]).reshape(-1,1)
    
def learn_PQ(rank, model_name, epochs, R):
    n_p, n_q = R.shape

    obs = ~np.isnan(R)
    ts_R = torch.Tensor(R[obs].reshape(-1,1))

    Omega = np.array([np.where(obs)[0], np.where(obs)[1], np.arange(obs.sum())]).T
    n_Omega = len(Omega)

    Omega_tr, Omega_val = torch.utils.data.random_split(Omega, [int(n_Omega * 0.9), n_Omega-int(n_Omega * 0.9)])
    dl_Omega_tr = torch.utils.data.DataLoader(Omega_tr, batch_size = 100)
    dl_Omega_val = torch.utils.data.DataLoader(Omega_val, batch_size = 100)

    if model_name == "LN" :
        model = LN_torch(n_p, n_q, rank)
    else :
        model = MIRT_torch(n_p, n_q, rank)

    optimizer = torch.optim.SparseAdam(model.parameters())
    loss_func = torch.nn.BCELoss()
    P = model.p.weight.clone().detach()
    Q = model.q.weight.clone().detach()

    loss_val_min = 1e9
    pbar = tqdm(range(epochs + 1))
    for epoch in pbar:
        
        for omega in dl_Omega_tr:
            pred = model(omega.T[0], omega.T[1])
            loss = loss_func(pred, ts_R[omega.T[2]])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for omega in dl_Omega_val:
            pred_val = model(omega.T[0], omega.T[1])
            loss_val = loss_func(pred_val, ts_R[omega.T[2]])
        
        if loss_val_min > loss_val.item():
            loss_val_min = loss_val.item()
            P = model.p.weight.clone().detach()
            Q = model.q.weight.clone().detach()

        pbar.set_description("iteration : {} / train loss : {} / valid loss : {} / min loss : {}".format(epoch, loss.item(), loss_val.item(), loss_val_min))

    return P, Q
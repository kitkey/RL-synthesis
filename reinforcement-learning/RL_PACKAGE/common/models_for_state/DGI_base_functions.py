import os
import pathlib
from math import ceil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import xavier_normal
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

import igraph
import pyintergraph
from category_encoders import HelmertEncoder, OneHotEncoder


import copy
from typing import Callable, Tuple

from torch_geometric.nn.inits import reset, uniform, glorot
import random

EPS = 1e-15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12 if DEVICE == "cuda" else 0

def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



scheme_list = ['bp_be', 'spi', 'ss_pcm', 'usb_phy', 'des3_area', 'fpu', 'aes_xcrypt',
                            'tinyRocket', 'pci', 'simple_spi', 'aes', 'wb_dma', 'vga_lcd', 'fir',
                            'tv80', 'aes_secworks', 'dynamic_node', 'sha256', 'ac97_ctrl', 'i2c',
                            'ethernet', 'mem_ctrl', 'iir', 'sasc', 'wb_conmax', 'picosoc']


class DeepGraphInfomax(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_weights: int,
            encoder: Module,
            summary: Callable,
            corruption: Callable,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.empty(hidden_weights, hidden_weights))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        xavier_normal(self.weight)

    def forward(self, *args, x_corrupt=None, edge_corrupt=None, batch_corrupt=None, **kwargs) -> Tuple[
        Tensor, Tensor, Tensor]:
        pos_z = self.encoder(*args, **kwargs)
        neg_z = 0
        if x_corrupt is not None:
            cor = self.corruption(x_corrupt, edge_corrupt, batch_corrupt)
            cor = cor if isinstance(cor, tuple) else (cor,)
            cor_args = cor[:len(args)]
            cor_kwargs = copy.copy(kwargs)
            for key, value in zip(kwargs.keys(), cor[len(args):]):
                cor_kwargs[key] = value

            neg_z = self.encoder(*cor_args, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary

    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(
            self,
            train_z: Tensor,
            train_y: Tensor,
            test_z: Tensor,
            test_y: Tensor,
            solver: str = 'lbfgs',
            multi_class: str = 'auto',
            *args,
            **kwargs,
    ) -> float:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_weights):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_weights)
        ])
        self.W = nn.Linear(4, hidden_channels, device=DEVICE)
        self.W1 = Parameter(torch.empty((hidden_channels, 4)))
        # xavier_normal(self.W1)
        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Identity()
        ])


    def forward(self, x, edge_index, batch_size):
        i = 0
        x_theta = torch.matmul(x, self.W1.T)
        h = 0
        for conv, act in zip(self.convs, self.activations):
            if i > 0:
                x = conv(h + x_theta, edge_index)
                x = act(x)
                if i < 2:
                    h += x
            else:
                x = conv(x, edge_index)
                x = act(x)
                h += x
            i += 1

        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x, edge_index, batch_size


def get_scheme_paths():
    steps = [i for i in range(1, 21)]
    syns = [0, 1, 2, 3]
    scheme_paths = [pathlib.Path(f"graphml_schemes/{name}_list_syns/syn{j}/{name}_syn{j}_step{z}.graphml") for name in
                    scheme_list for z in random.sample(steps, k=2) for j in random.sample(syns, k=2)]
    return scheme_paths

def _init_fn(worker_id):
    return np.random.seed(42)
def graph_load(graph_path, pos_size=None, pos_batch_size=None):
    # seed_torch()
    torch.cuda.manual_seed(123)
    torch.manual_seed(123)
    device = "cuda"
    gi = igraph.read(graph_path)  # "picosoc_step0.graphml"
    g = pyintergraph.igraph2nx(gi)
    tgg = torch_geometric.utils.from_networkx(g)
    node_type_ = pd.DataFrame(tgg["node_type"].numpy(), columns=["categ"])
    enc = OneHotEncoder(return_df=False, cols=["categ"]).fit(node_type_)
    node_type_ = torch.tensor(enc.transform(node_type_))
    tgg.x = torch.cat((node_type_, torch.tensor(tgg["num_inverted_predecessors"]).unsqueeze(-1)), dim=-1).to(
        torch.float32)

    data = tgg.to(device, 'x', 'edge_index')
    batch_size = 128
    if pos_size is not None:
        batch_size = ceil(data["num_nodes"] / pos_size * pos_batch_size)

    del data["num_inverted_predecessors"]
    del data["node_id"]
    del data["id"]
    del data["edge_type"]
    del data["num_nodes"]
    del data["node_type"]
    train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=batch_size,
                                  shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=_init_fn)
    return train_loader


def networkx_load(G, device):
    # seed_torch()
    torch.cuda.manual_seed(123)
    torch.manual_seed(123)
    tgg = torch_geometric.utils.from_networkx(G)
    node_type_ = pd.DataFrame(tgg["node_type"].numpy(), columns=["categ"])
    enc = OneHotEncoder(return_df=False, cols=["categ"]).fit(node_type_)
    node_type_ = torch.tensor(enc.transform(node_type_))
    tgg.x = torch.cat((node_type_, torch.tensor(tgg["num_inverted_predecessors"]).unsqueeze(-1)), dim=-1).to(
        torch.float32)
    data = tgg.to(device, 'x', 'edge_index')
    data.x = data.x.float()
    batch_size = 128

    del data["num_inverted_predecessors"], data["node_id"], data["id"], data["edge_type"], data["num_nodes"], \
        data["node_type"]

    train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=_init_fn)
    return train_loader



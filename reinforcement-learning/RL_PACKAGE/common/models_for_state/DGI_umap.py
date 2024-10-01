import pathlib
import random

import numpy as np
import umap
import umap.plot
import pandas as pd
import torch
from torch import nn

from __init__ import PATH_TO_SAVE_DGI, PATH_TO_DGI_CSV
from DGI_base_functions import DeepGraphInfomax, Encoder, corruption, graph_load

import warnings

warnings.filterwarnings("ignore")


scheme_list = ['bp_be', 'spi', 'ss_pcm', 'usb_phy', 'des3_area', 'fpu', 'aes_xcrypt',
               'tinyRocket', 'pci', 'simple_spi', 'aes', 'wb_dma', 'vga_lcd', 'fir',
               'tv80', 'aes_secworks', 'dynamic_node', 'sha256', 'ac97_ctrl', 'i2c',
               'ethernet', 'mem_ctrl', 'iir', 'sasc', 'wb_conmax', 'picosoc']


def generate_dataset_umap():
    device = "cuda"
    syns = [0, 1, 2, 3]
    steps = [i for i in range(1, 21)]

    paths = [[pathlib.Path(f"graphml_schemes/{name}_list_syns/syn{j}/{name}_syn{j}_step{z}.graphml"), name] for name in
             scheme_list
             for z in random.sample(steps, k=15) for j in random.sample(syns, k=3)]
    hidden_weights = 16
    hidden_channels = 512
    model = DeepGraphInfomax(
        hidden_channels=hidden_channels, encoder=Encoder(4, hidden_channels, hidden_weights),  # (2, 1024)
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption, hidden_weights=hidden_weights).to(device)

    model.load_state_dict(state_dict=torch.load(PATH_TO_SAVE_DGI))
    model = model.to(device)

    @torch.no_grad()
    def test(trainer):
        model.eval()

        zs = []
        for batch in (trainer):
            pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
            zs.append(pos_z.cpu())
        z = torch.cat(zs, dim=0)
        return z

    s = nn.Sigmoid()

    data = pd.DataFrame(columns=[i for i in range(1, 17)] + ["name"])
    c = 0
    for path in paths:
        print(path)
        try:
            trainer = s(test(graph_load(path[0])).mean(dim=0)).numpy()
        except:
            c += 1
            print("c ", c)
            continue

        data.loc[len(data.index)] = [i for i in trainer] + [path[1]]
    data.to_csv(path_or_buf=PATH_TO_DGI_CSV)


class UmapObjectPainter:
    def __init__(self, embedding, data_targets, data_train=None, classes_list_buffer=None):
        self.embedding = embedding
        self.data_train = data_train
        self.data_targets = data_targets
        self.classes_list_buffer = classes_list_buffer

    def umap_embeds_painter(self) -> None:
        s = umap.plot.points(self.embedding, labels=self.data_targets, cmap='RdYlBu_r', theme='fire', width=1300)
        umap.plot.show(s)

    def interactive_painter(self) -> None:
        hover_data = pd.DataFrame({'index': np.arange(len(self.data_targets)), 'label': self.data_targets})
        hover_data['item'] = hover_data.label.map(self.classes_list_buffer)

        p = umap.plot.interactive(self.embedding, labels=self.data_targets, hover_data=hover_data, point_size=5,
                                  cmap='RdYlBu_r',
                                  tools=["pan", "wheel_zoom", "box_zoom", "save", "reset", "help", ])
        umap.plot.show(p)


data = pd.read_csv(PATH_TO_DGI_CSV)

classes_list_buffer = {j + 1: data['name'][j] for j in range(data.shape[0])}

calls = data['name'].copy()
data.drop(columns=['name'], inplace=True)
embed_space = umap.UMAP(n_neighbors=7, min_dist=0.7, metric='mahalanobis', init='pca', random_state=42,
                        repulsion_strength=0.4, negative_sample_rate=15, verbose=True).fit(data)

kt = UmapObjectPainter(embed_space, calls, data, classes_list_buffer)
kt.interactive_painter()

import random

import torch
from tqdm import tqdm

from __init__ import PATH_TO_SAVE_DGI
from DGI_base_functions import DeepGraphInfomax, Encoder, corruption, graph_load, \
    get_scheme_paths

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


def main_training(epochs):
    """
    Для начала обучения нужны graphml файлы формата "graphml_schemes/{name}_list_syns/syn{j}/{name}_syn{j}_step{z}.graphml"
    """
    hidden_weights = 16
    hidden_channels = 1024
    epochs = 20
    model = DeepGraphInfomax(
        hidden_channels=hidden_channels, encoder=Encoder(4, hidden_channels, hidden_weights),  # (2, 1024)
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption, hidden_weights=hidden_weights).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
    scheme_paths = get_scheme_paths()

    def train(epoch):
        model.train()
        total_loss = total_examples = 0
        loss_i = 1e+8
        for i in range(epochs):

            j = 0
            path_pos, path_neg = random.sample(scheme_paths, 2)
            try:
                print(path_pos, path_neg)
                pos_train_loader = graph_load(path_pos)
                neg_train_loader = graph_load(path_neg)
            except:
                print("error")
                continue

            for pos_batch, neg_batch in tqdm(zip(pos_train_loader, neg_train_loader), desc=f'Epoch {epoch:02d}'):
                j += 1
                optimizer.zero_grad()
                pos_z, neg_z, summary = model(pos_batch.x, pos_batch.edge_index, pos_batch.batch_size,
                                              x_corrupt=neg_batch.x, edge_corrupt=neg_batch.edge_index,
                                              batch_corrupt=neg_batch.batch_size)
                loss = model.loss(pos_z, neg_z, summary)
                loss_i += loss
                print(loss)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pos_z.size(0)
                total_examples += pos_z.size(0)
        return total_loss / 30

    sum_loss_deeper = train(epochs)
    torch.save(model.state_dict(), f=PATH_TO_SAVE_DGI)
    return sum_loss_deeper

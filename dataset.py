import numpy as np
from torch.utils.data import Dataset
import torch
from utils import *


class ClsDataset(Dataset):
    def __init__(self, inputs, targets, device):
        self.inputs = torch.from_numpy(inputs).float().to(device)
        self.targets = torch.from_numpy(targets).float().to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]


class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer, token_len, device, text_mode, datafile):
        if dataset == 'essays':
            author_ids, input_ids, targets = essays_embeddings(
                datafile, tokenizer, token_len, text_mode
            )
        elif dataset == 'kaggle':
            pass

        author_ids = torch.from_numpy(np.array(author_ids)).long().to(device)
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(device)
        targets = torch.from_numpy(np.array(targets))

        self.author_ids = author_ids
        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.author_ids[item], self.input_ids[item], self.targets[item]


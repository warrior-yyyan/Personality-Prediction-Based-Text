import os
import sys
import numpy as np
import re
import pickle
from sklearn.model_selection import StratifiedKFold
import utils
import keras
import models.MLP as MLP
import torch.nn as nn
import torch.optim as optim
from dataset import ClsDataset
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_inputs(inp_dir, dataset, embed_model, embed_mode, text_mode, n_layer):
    
    file = open(
        inp_dir + dataset + '-' + embed_model + '-' + embed_mode + '-' + text_mode + '.pkl', 'rb'
    )
    data = pickle.load(file)

    # data_x (n_batch, hidden_layer, batch_size, hidden_dim)  deberta-large->(20, 24, 128, 1024)
    # data_y (n_batch, batch_size, n_classes) (20, 128, 5)
    author_ids, data_x, data_y = list(zip(*data))

    file.close()

    mask_w = np.zeros([n_hl])
    mask_w[int(n_layer) - 1] = 1

    inputs = []
    targets = []
    n_batch = len(data_y)

    for i in range(n_batch):
        inputs.extend(np.einsum('k, kij->ij', mask_w, data_x[i]))
        targets.extend(data_y[i])
        
    inputs = np.array(inputs)
    full_targets = np.array(targets)
    
    return inputs, full_targets


def train(dataset, inputs, full_targets):
    
    if dataset == 'kaggle':
        trait_labels = ['E', 'N', 'F', 'J']
    else:
        trait_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

    n_splits = 10
    expdata = {'acc': [], 'trait': [], 'fold': []}

    for trait_idx in range(full_targets.shape[1]):
        targets = full_targets[:, trait_idx]

        expdata['trait'].extend([trait_labels[trait_idx]] * n_splits)
        expdata['fold'].extend(np.arange(1, n_splits + 1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

        for train_index, test_index in skf.split(inputs, targets):
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]

            # converting to one-hot embedding
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

            model = MLP.Model(hidden_dim, 512, n_classes)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # train
            train_dataset = ClsDataset(x_train, y_train, device)
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            model.train()
            for epoch in range(epochs):
                tol_loss = 0
                for i, (trains, targets) in enumerate(train_dataloader, 0):
                    outpus = model(trains)
                    optimizer.zero_grad()
                    loss = criterion(outpus, targets)
                    loss.backward()
                    optimizer.step()

                    tol_loss += loss.item()

                print(f'epoch : {1 + epoch}, loss : {tol_loss}')





    


if __name__ == '__main__':
    (
        inp_dir,
        dataset,
        lr,
        batch_size,
        epochs,
        embed_model,
        n_layer,
        text_mode,
        embed_mode,
        seedid,
    ) = utils.parse_args_classifier()

    print('{} : {} : {} : {} : {} : {} : {} : {} : {}'.format(dataset, lr, batch_size, epochs,
                                                              embed_model, n_layer, text_mode,
                                                              embed_mode, seedid))
    n_classes = 2
    np.random.seed(seedid)
    out_path = '/output/'

    if re.search(r'base', embed_model):
        n_hl = 12
        hidden_dim = 768

    elif re.search(r'large', embed_model):
        n_hl = 24
        hidden_dim = 1024
        
    inputs, full_targets = get_inputs(inp_dir, dataset, embed_model, embed_mode, text_mode, n_layer)
    df = train(dataset, inputs, full_targets)
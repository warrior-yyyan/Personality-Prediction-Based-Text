import os
import sys
import numpy as np
import re
import pickle
from sklearn.model_selection import StratifiedKFold
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ClsDataset
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from importlib import import_module
import torch.nn.functional as F
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

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

    final_acc = []

    for trait_idx in range(full_targets.shape[1]):
        targets = full_targets[:, trait_idx]

        expdata['trait'].extend([trait_labels[trait_idx]] * n_splits)
        expdata['fold'].extend(np.arange(1, n_splits + 1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        k = 0
        tot_acc = []
        tot_loss = []
        for train_index, test_index in skf.split(inputs, targets):
            k += 1
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            # converting to one-hot embedding
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

            x = import_module('models.' + ft_model)
            model = x.Model(hidden_dim, n_classes).to(device)
            # utils.init_network(model)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # prepare data
            train_dataset = ClsDataset(x_train, y_train, device)
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            test_dataset = ClsDataset(x_test, y_test, device)
            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            # train
            acc_list = []
            loss_list = []
            report_list = []
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for i, (trains, labels) in enumerate(train_dataloader, 0):
                    outpus = model(trains)
                    optimizer.zero_grad()
                    loss = criterion(outpus, labels)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                test_acc, test_loss, test_report, test_confusion = evaluate(model, test_dataloader)
                acc_list.append(test_acc)
                loss_list.append(test_loss)
                report_list.append(test_report)

                # print(f'Epoch : {1 + epoch} -> Train Loss : {train_loss / len(train_dataloader)}, '
                #       f'Test Acc : {test_acc}, Test Loss : {test_loss}')

            best_acc = max(acc_list)
            best_loss = loss_list[acc_list.index(max(acc_list))]
            best_report = report_list[acc_list.index(max(acc_list))]

            print(f'Trait : {trait_labels[trait_idx]}, No.{k} fold, Best Acc : {best_acc}, Best Loss : {best_loss}')

            tot_acc.append(best_acc)
            tot_loss.append(best_loss)

        final_acc.append(sum(tot_acc) / n_splits)
        print('='*100)
        print(f'Trait : {trait_labels[trait_idx]}, {n_splits} fold Avg.Acc : {sum(tot_acc) / n_splits}, {n_splits} fold Avg.Loss : {sum(tot_loss) / n_splits}')
        print('='*100)

    print(f'Five Train Avg.Acc : {sum(final_acc) / len(final_acc)}')


def evaluate(model, data):
    model.eval()
    test_loss = 0
    true_all = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for i, (tests, labels) in enumerate(data):
            out = model(tests)
            loss = F.binary_cross_entropy_with_logits(out, labels)
            test_loss += loss
            true = torch.tensor(np.array([np.argmax(i) for i in labels.cpu().numpy()]), dtype=torch.int64)
            predict = torch.max(out.data, 1)[1].cpu()
            true_all = np.append(true_all, true)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(true_all, predict_all)
    report = metrics.classification_report(true_all, predict_all, target_names=['not', 'is'], digits=4)
    confusion = metrics.confusion_matrix(true_all, predict_all)

    return acc, test_loss / len(data), report, confusion


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
        ft_model,
        jobid,
    ) = utils.parse_args_classifier()

    print(
        'dataset:{} | lr:{} | batch_size:{} | epochs:{} | embed_model:{} | n_layer:{} | text_mode:{} | embed_mode:{} | seedid:{}'.format(
            dataset, lr, batch_size, epochs,
            embed_model, n_layer, text_mode,
            embed_mode, jobid))

    n_classes = 2
    utils.setup_seed(jobid)

    out_path = '/output/'

    if re.search(r'base', embed_model):
        n_hl = 12
        hidden_dim = 768

    elif re.search(r'large', embed_model):
        n_hl = 24
        hidden_dim = 1024
        
    inputs, full_targets = get_inputs(inp_dir, dataset, embed_model, embed_mode, text_mode, n_layer)
    train(dataset, inputs, full_targets)

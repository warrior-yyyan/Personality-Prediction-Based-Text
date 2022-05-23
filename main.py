import os
import sys
import numpy as np
import pickle
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *

from dataset import MyDataset
from utils import *

sys.path.insert(0, os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(embed_model, local_model_path=None, local_tokenizer_path=None):
    """
    Load model and tokenizer by huggingface.
    If you have not internet,
    you could download the offline file and use param of 'local_model_path' and 'local_tokenizer_path'
    :param embed_model: 'bert-base'
    :param local_model_path: '/model/yyyan/bert-base-uncased/base'
    :param local_tokenizer_path: '/model/yyyan/bert-base-uncased/base/'
    :return: model, tokenizer, n_hl, hidden_dim
    """
    # * Model          | Tokenizer          | Pretrained weights shortcut
    # MODEL=(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')

    if embed_model == 'bert-base':
        n_hl = 12
        hidden_dim = 768
        MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')

    elif embed_model == 'bert-large':
        n_hl = 24
        hidden_dim = 1024
        MODEL = (BertModel, BertTokenizer, 'bert-large-uncased')

    elif embed_model == 'roberta-base':
        n_hl = 12
        hidden_dim = 768
        MODEL = (RobertaModel, RobertaTokenizer, 'roberta-base')

    elif embed_model == 'roberta-large':
        n_hl = 24
        hidden_dim = 1024
        MODEL = (RobertaModel, RobertaTokenizer, 'roberta-large')

    elif embed_model == 'deberta-v3-base':
        n_hl = 12
        hidden_dim = 768
        MODEL = (DebertaV2Model, DebertaV2Tokenizer, 'deberta-v3-base')

    elif embed_model == 'deberta-v3-large':
        n_hl = 24
        hidden_dim = 1024
        MODEL = (AutoModel, AutoTokenizer, 'deberta-v3-large')

    elif embed_model == 'albert-base':
        n_hl = 12
        hidden_dim = 768
        MODEL = (AlbertModel, AlbertTokenizer, 'albert-base-v2')

    elif embed_model == 'albert-large':
        n_hl = 24
        hidden_dim = 1024
        MODEL = (AlbertModel, AlbertTokenizer, 'albert-large-v2')

    model_class, tokenizer_class, pretrained_weights = MODEL

    # load the LM model and tokenizer from the HuggingFace Transformers library
    # use offline file
    model = model_class.from_pretrained(
        local_model_path, output_hidden_states=True
    )

    # output_attentions=False
    tokenizer = tokenizer_class.from_pretrained(
        local_tokenizer_path, do_lower_case=True
    )

    return model, tokenizer, n_hl, hidden_dim


def extract_features(input_ids, text_mode, n_hl):
    """Extract bert embedding for each input."""
    tmp = []
    output = model(input_ids)
    # Bert output detail : https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertModel
    # Deberta-v2/3 : https://huggingface.co/docs/transformers/main/en/model_doc/deberta-v2#transformers.DebertaV2Model

    for i in range(n_hl):
        if embed_mode == "cls":
            # bert : tmp.append(output[2][i + 1][:, 0, :].cpu().numpy())
            # deberta-v2/3 : tmp.append(output[1][i + 1][:, 0, :].cpu().numpy())

            tmp.append(output[1][i + 1][:, 0, :].cpu().numpy())
            print(output.shape)

        elif embed_mode == "mean":
            # bert : tmp.append((output[2][i + 1].cpu().numpy()).mean(axis=1))
            # deberta-v2/3 : tmp.append((output[1][i + 1].cpu().numpy()).mean(axis=1))

            tmp.append((output[1][i + 1].cpu().numpy()).mean(axis=1))
            print(output.shape)

    hidden_features.append(np.array(tmp))


if __name__ == '__main__':
    (
        dataset,
        token_len,
        batch_size,
        embed_model,
        out_dir,
        text_mode,
        embed_mode,
        local_model_path,
        local_tokenizer_path,
    ) = parse_args_main()
    print(
        '{} | {} | {} | {} | {} | {} | {} | {} | {}'.format(dataset, token_len, batch_size,
                                                            embed_model, out_dir, text_mode, embed_mode,
                                                            local_model_path, local_tokenizer_path)
    )

    # n_hl : num of hidden layer
    model, tokenizer, n_hl, hidden_dim = get_model(embed_model, local_model_path, local_tokenizer_path)

    datafile = 'data/essays/essays.csv' if dataset == 'essays' else 'data/kaggle/kaggle.csv'

    dataset_ = MyDataset(dataset, tokenizer, token_len, device, text_mode, datafile)
    dataloader_ = DataLoader(
        dataset=dataset_,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("starting to extract Pretrained Language Model embeddings...")

    model.to(device)

    hidden_features = []
    all_targets = []
    all_author_ids = []

    # get bert embedding for each input
    for author_ids, input_ids, targets in dataloader_:
        with torch.no_grad():
            all_targets.append(targets.cpu().numpy())
            all_author_ids.append(author_ids.cpu().numpy())
            extract_features(input_ids, text_mode, n_hl)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pkl_file_name = dataset + "-" + embed_model + "-" + embed_mode + "-" + text_mode + ".pkl"

    file = open(os.path.join(out_dir, pkl_file_name), 'wb')
    pickle.dump(zip(all_author_ids, hidden_features, all_targets), file)
    file.close()

    print(f'extracting embeddings for {dataset} dataset: DONE!')


import numpy as np
import argparse
import pandas as pd
import re
import csv
import preprocessor as p
import torch
import random
import torch
import torch.nn as nn

def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r'http\S+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\|\|\|', ' ', sentence)
    return sentence


def load_essays(datafile):
    with open(datafile, 'rt') as csvf:
        lines = csv.reader(csvf, delimiter=',', quotechar='"')
        flag = True
        df = pd.DataFrame(
            columns=['user', 'text', 'token_len', 'EXT', 'NEU', 'AGR', 'CON', 'OPN']
        )
        for line in lines:
            if flag:
                flag = False
                continue

            df = df.append(
                {
                    'user': line[0],
                    'text': line[1],
                    'token_len': 0,
                    'EXT': 1 if line[2].lower() == 'y' else 0,
                    'NEU': 1 if line[3].lower() == 'y' else 0,
                    'AGR': 1 if line[4].lower() == 'y' else 0,
                    'CON': 1 if line[5].lower() == 'y' else 0,
                    'OPN': 1 if line[6].lower() == 'y' else 0,
                },
                ignore_index=True,
            )

    print('EXT : ', df['EXT'].value_counts())
    print('NEU : ', df['NEU'].value_counts())
    print('AGR : ', df['AGR'].value_counts())
    print('CON : ', df['CON'].value_counts())
    print('OPN : ', df['OPN'].value_counts())

    return df


def essays_embeddings(datafile, tokenizer, token_len, text_mode):
    targets = []
    input_ids = []
    df = load_essays(datafile)

    for i in df.index:
        tokens = tokenizer.tokenize(df['text'][i])
        df.at[i, 'token_len'] = len(tokens)

    df.sort_values(by=['token_len', 'user'], inplace=True, ascending=True)
    tmp_df = df['user']
    tmp_df.to_csv('data/author_id_order.csv', index_label='order')

    print(f"essays: min_token {df['token_len'].min()}, max_token {df['token_len'].max()}, "
          f"mean_token {df['token_len'].mean()}")

    for i in range(len(df)):
        text = preprocess_text(df['text'][i])
        tokens = tokenizer.tokenize(text)

        if text_mode == 'normal' or text_mode == '512_head':
            input_ids.append(
                tokenizer.encode(
                    tokens,
                    add_special_tokens=True,
                    max_length=token_len,
                    pad_to_max_length=True,
                )
            )
        elif text_mode == '512_tail':
            input_ids.append(
                tokenizer.encode(
                    tokens[-(token_len - 2):],
                    add_special_tokens=True,
                    max_length=token_len,
                    pad_to_max_length=True,
                )
            )
        elif text_mode == "256_head_tail":
            input_ids.append(
                tokenizer.encode(
                    tokens[: (token_len - 1)] + tokens[-(token_len - 1):],
                    add_special_tokens=True,
                    max_length=token_len,
                    pad_to_max_length=True,
                )
            )

        targets.append(
            [df['EXT'][i], df['NEU'][i], df['AGR'][i], df['CON'][i], df['OPN'][i]]
        )

    author_ids = np.array(df.index)
    print("loaded all input_ids and targets from the data file!")
    return author_ids, input_ids, targets


def parse_args_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='essays')
    ap.add_argument('--token_len', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--embed_model', type=str, default='deberta-v3-large')
    ap.add_argument('--out_dir', type=str, default='/output/')
    ap.add_argument('--text_mode', type=str, default='256_mean_tail')
    ap.add_argument('--embed_mode', type=str, default='mean')
    ap.add_argument('--local_model_path', type=str, default='/model/yyyan/deberta-v3-large')
    ap.add_argument('--local_tokenizer_path', type=str, default='/model/yyyan/deberta-v3-large/')
    args = ap.parse_args()
    return (
        args.dataset,
        args.token_len,
        args.batch_size,
        args.embed_model,
        args.out_dir,
        args.text_mode,
        args.embed_mode,
        args.local_model_path,
        args.local_tokenizer_path,
    )


def parse_args_classifier():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp_dir', type=str, default='data/essays/')
    ap.add_argument('--dataset', type=str, default='essays')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--embed_model', type=str, default='deberta-v3-large')
    # best layer 13 ~ 14
    ap.add_argument('--n_layer', type=str, default='12')
    ap.add_argument('--text_mode', type=str, default='256_head_tail')
    ap.add_argument('--embed_mode', type=str, default='mean')
    ap.add_argument('--ft_model', type=str, default='MLP')
    ap.add_argument('--jobid', type=int, default=0)
    args = ap.parse_args()
    return (
        args.inp_dir,
        args.dataset,
        args.lr,
        args.batch_size,
        args.epochs,
        args.embed_model,
        args.n_layer,
        args.text_mode,
        args.embed_mode,
        args.ft_model,
        args.jobid,
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_network(model, method='xavier'):
    for name, w in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass










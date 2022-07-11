import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p


def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r'http\S+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    # remove kaggle.csv '|||'
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


def load_kaggle_df(datafile):
    with open(datafile, "rt", encoding="utf-8") as csvf:
        csvreader = csv.reader(csvf, delimiter=",", quotechar='"')
        first_line = True
        df = pd.DataFrame(columns=["user", "text", "E", "N", "F", "J"])
        for line in csvreader:
            if first_line:
                first_line = False
                continue

            text = line[1]

            df = df.append(
                {
                    "user": line[3],
                    "text": text,
                    "E": 1 if line[0][0] == "E" else 0,
                    "N": 1 if line[0][1] == "N" else 0,
                    "F": 1 if line[0][2] == "F" else 0,
                    "J": 1 if line[0][3] == "J" else 0,
                },
                ignore_index=True,
            )

    print("E : ", df["E"].value_counts())
    print("N : ", df["N"].value_counts())
    print("F : ", df["F"].value_counts())
    print("J : ", df["J"].value_counts())

    return df


def kaggle_embeddings(datafile, tokenizer, token_length):
    hidden_features = []
    targets = []
    token_len = []
    input_ids = []
    author_ids = []

    df = load_kaggle_df(datafile)
    cnt = 0
    for ind in df.index:

        text = preprocess_text(df["text"][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        token_ids = tokenizer.encode(
            tokens,
            add_special_tokens=True,
            max_length=token_length,
            pad_to_max_length=True,
        )
        if cnt < 10:
            print(tokens[:10])

        input_ids.append(token_ids)
        targets.append([df["E"][ind], df["N"][ind], df["F"][ind], df["J"][ind]])
        author_ids.append(int(df["user"][ind]))
        cnt += 1

    print("average length : ", int(np.mean(token_len)))
    author_ids = np.array(author_ids)

    return author_ids, input_ids, targets

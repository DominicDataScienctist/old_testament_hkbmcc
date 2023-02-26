import argparse

import numpy as np
import requests

import pandas as pd
import yaml

from utils.preprocess import acquire_data


def convert_data_spreadsheet(data_dir, tfidf_fn, df_vocab_fn, df_fn, file_type='csv'):
    tfidf_df = pd.read_pickle(f'{data_dir}/{tfidf_fn}')
    df_vocab = pd.read_pickle(f'{data_dir}/{df_vocab_fn}')
    df = pd.read_pickle(f'{data_dir}/{df_fn}')

    tfidf_spreadsheet_fn_str = tfidf_fn.split('.')[0]
    df_vocab_spreadsheet_fn_str = df_vocab_fn.split('.')[0]
    df_spreadsheet_fn_str = df_fn.split('.')[0]

    tfidf_df.to_csv(f"{tfidf_spreadsheet_fn_str}.csv")
    df_vocab.to_csv(f"{df_vocab_spreadsheet_fn_str}.csv")
    df.to_csv(f"{df_spreadsheet_fn_str}.csv")


def query(payload, api_url, headers):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


def get_connection_config(token):
    headers = {"Authorization": f"Bearer {token}"}
    return headers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['acquisition', 'export', 'summarize', 'index'], default='export')
    parser.add_argument('--bible_url', type=str, default='https://bereanbible.com/bsb.txt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--tfidf_fn', type=str, default='tfidf_df.pkl')
    parser.add_argument('--df_vocab_fn', type=str, default='df_vocab.pkl')
    parser.add_argument('--df_fn', type=str, default='original_data.pkl')
    parser.add_argument('--config', type=str, default='config.yaml')
    # parser.add_argument('index_chapter', type=str, default='chap')
    args = parser.parse_args()
    task = args.task

    if task == 'acquisition':
        bible_url = args.bible_url

        data_dir = args.data_dir
        tfidf_fn = args.tfidf_fn
        df_vocab_fn = args.df_vocab_fn
        df_fn = args.df_fn
        acquire_data(data_dir, bible_url, tfidf_fn, df_vocab_fn, df_fn)

    elif task == 'export':
        data_dir = args.data_dir
        tfidf_fn = args.tfidf_fn
        df_vocab_fn = args.df_vocab_fn
        df_fn = args.df_fn
        convert_data_spreadsheet(data_dir, tfidf_fn, df_vocab_fn, df_fn)

    elif task == 'summarize':
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        token = config['hugging_face_token']
        df_fn = args.df_fn
        df = pd.read_pickle(f"data/{df_fn}")
        df_chapter = df[['book', 'chapter', 'text']].groupby(['book', 'chapter'])['text'].apply(lambda x: ' '.join(x))
        df_chapter = df_chapter.reset_index()
        df_book = df[['book', 'text']].groupby(['book'])['text'].apply(lambda x: ' '.join(x))
        df_book = df_book.reset_index()
        haggai = str(df_book.loc[df_book['book'] == 'Haggai'].text.values[0])
        api_url = config['api_url']
        headers = get_connection_config(token)
        output = query({"inputs": haggai, "max_length":5}, api_url=api_url, headers=headers)
        print(output)

    elif task == 'index':
        df_fn = args.df_fn
        df = pd.read_pickle(f"data/{df_fn}")
        print(np.unique(df['book']))


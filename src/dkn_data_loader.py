# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/hwwang55/DKN/blob/master/src/data_loader.py

# import libraries
import pandas as pd
import numpy as np
from collections import namedtuple
from typing import Tuple, NamedTuple, Dict
from pathlib import Path

# import custom code
from src.config import FILENAME_TRAIN_DATA, FILENAME_TEST_DATA, FILENAME_RANDOM_TEST_DATA

Data = namedtuple('Data', ['size', 'clicked_words', 'clicked_entities', 'news_words', 'news_entities', 'labels'])


def load_data(args) -> Tuple[NamedTuple, NamedTuple, NamedTuple]:
    """ 
    Loads the data for the DKN model. 

    Returns:
        - :obj:`NamedTuple`:
            Train data.
        - :obj:`NamedTuple`:
            Vaidation data.
        - :obj:`NamedTuple`:
            Test data.
    """
    train_df = read(FILENAME_TRAIN_DATA)
    test_df = read(FILENAME_TEST_DATA)
    random_test_df = read(FILENAME_RANDOM_TEST_DATA)
    uid2words, uid2entities = aggregate(train_df, args.max_click_history)
    all_train_data = transform(train_df, uid2words, uid2entities)
    test_data = transform(test_df, uid2words, uid2entities)
    random_test_data = transform(random_test_df, uid2words, uid2entities)

    # Split train data into train and validation sets
    val_ratio = 0.2
    all_train_indices = np.array(np.arange(0, all_train_data.size))
    val_indices = np.random.choice(all_train_indices, size=int(all_train_indices.shape[0] * val_ratio), replace=False)
    train_indices = np.asarray(list(set(all_train_indices) - set(val_indices)))

    val_data = Data(
        size=val_indices.shape[0],
        clicked_words = all_train_data.clicked_words[val_indices],
        clicked_entities = all_train_data.clicked_entities[val_indices],
        news_words = all_train_data.news_words[val_indices],
        news_entities = all_train_data.news_entities[val_indices],
        labels = all_train_data.labels[val_indices]
        )
    train_data = Data(
        size=val_indices.shape[0],
        clicked_words = all_train_data.clicked_words[train_indices],
        clicked_entities = all_train_data.clicked_entities[train_indices],
        news_words = all_train_data.news_words[train_indices],
        news_entities = all_train_data.news_entities[train_indices],
        labels = all_train_data.labels[train_indices]
        )

    return train_data, val_data, test_data, random_test_data


def read(file: Path) -> pd.DataFrame:
    df = pd.read_table(file, sep='\t', header=None, names=['user_id', 'news_words', 'news_entities', 'label'])
    df['news_words'] = df['news_words'].map(lambda x: [int(i) for i in x.split(',')])
    df['news_entities'] = df['news_entities'].map(lambda x: [int(i) for i in x.split(',')])
    return df


def aggregate(train_df: pd.DataFrame, max_click_history: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    uid2words = dict()
    uid2entities = dict()
    pos_df = train_df[train_df['label'] == 1]
    for user_id in set(pos_df['user_id']):
        df_user = pos_df[pos_df['user_id'] == user_id]
        words = np.array(df_user['news_words'].tolist())
        entities = np.array(df_user['news_entities'].tolist())
        indices = np.random.choice(list(range(0, df_user.shape[0])), size=max_click_history, replace=True)
        uid2words[user_id] = words[indices]
        uid2entities[user_id] = entities[indices]
    return uid2words, uid2entities


def transform(df: pd.DataFrame, uid2words: Dict[int, int], uid2entities: Dict[int, int]) -> NamedTuple:
    df['clicked_words'] = df['user_id'].map(lambda x: uid2words[x])
    df['clicked_entities'] = df['user_id'].map(lambda x: uid2entities[x])
    data = Data(size=df.shape[0],
                clicked_words=np.array(df['clicked_words'].tolist()),
                clicked_entities=np.array(df['clicked_entities'].tolist()),
                news_words=np.array(df['news_words'].tolist()),
                news_entities=np.array(df['news_entities'].tolist()),
                labels=np.array(df['label']))
    return data

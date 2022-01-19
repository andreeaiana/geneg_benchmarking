# -*- coding: utf-8 -*-

""" Utility functions for preparing the data for DKN. """

# import libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# import custom code
from src.preprocess import preprocess
from src.config import FILENAME_TRAIN_RATINGS, FILENAME_TEST_RATINGS, FILENAME_NEWS_ID2INDEX, FILENAME_ENTITY_ID2INDEX, FILENAME_RAW_TRAIN_DATA, FILENAME_RAW_TEST_DATA, FILENAME_KG, FILENAME_SUBGRAPH, FILENAME_RAW_RANDOM_TEST_DATA, FILENAME_TEST_RATINGS_RANDOM
from src.config import DataFrameColumns
from src.fetch_data import get_raw_articles
from src.util.caching import load_cache, create_cache
from src.util.logger import setup_logging


def get_entities_per_article(news_id2index: Dict[int, str], entity_id2index: Dict[str, int], subkg: List) -> Dict[int, List[Tuple[int, str]]]:
    news2entities = defaultdict(list)
    for id in news_id2index.keys():
        news_node = news_id2index[id]
        direct_entities = [(entity_id2index[tail], tail) for head, _, tail in subkg if head==news_node]
        event_entities = [(entity_id2index[tail], tail) for head, _, tail in subkg if head==news_node + "_evt"]
        news2entities[id] = list(set(direct_entities).union(event_entities))
    return news2entities


def get_data_lines(data: np.array, articles: pd.DataFrame, news2entities: Dict[int, List[Tuple[int, str]]]) -> List: 
    lines = list()
    for item in tqdm(data):
        user = item[0]
        news_id = item[1]
        label = item[2]
        news_content = articles[DataFrameColumns.preprocessed_content].iloc[news_id]
        news_entities = news2entities[news_id]
        
        if news_entities:
            # Entities: entity_id_1:entity_name1;entity_id_2:entity_name_2;...
            entities = ";".join([str(ent[0]) + ":" + str(ent[1]) for ent in news_entities])

            # Line: user_id[TAB]news_title[TAB]label[TAB]entity_info
            line = str(user) + "\t" + str(news_content) + "\t" + str(label) + "\t" + entities + '\n'

            lines.append(line)
    return lines


def write_data_lines(file: Path, lines: List) -> None:
    with open(file, 'w') as f:
        f.writelines(lines)


def load_articles() -> pd.DataFrame:
    # Load articles
    articles = get_raw_articles()
    
    # Preprocess articles for Word2Vec
    logger.info('Preprocessing news content.')
    articles[DataFrameColumns.preprocessed_content] = articles[DataFrameColumns.content].apply(lambda x: preprocess(x))
    max_article_len = max(articles[DataFrameColumns.preprocessed_content].map(lambda x: len(x.split())))
    logger.info(f'Maximum article length: {max_article_len}.')

    return articles


def load_subgaph() -> List[int]:
    if os.path.isfile(FILENAME_SUBGRAPH):
        subgraph = load_cache(FILENAME_SUBGRAPH)
    else:
        # Load KG
        kg = load_cache(FILENAME_KG)

        # Create subgraph containing only the articles in the data and their entities
        logger.info('Creating subgraph.')
        events = [news + '_evt' for news in news_id2index.values()]
        subgraph = [triple for triple in tqdm(kg) if triple[0] in news_id2index.values() or triple[0] in events]
        
        # Cache subgraph
        create_cache(subgraph, FILENAME_SUBGRAPH)

    return subgraph

def create_raw_data_file(output_file: Path) -> None:
    # Get lines from the data
    logger.info('Preparing file lines.')
    lines = get_data_lines(data, articles, news2entities)

    # Write the lines to .txt file
    logger.info('Writing data to file.')
    write_data_lines(output_file, lines)
    logger.info('Finished.\n')


if __name__ == '__main__':
    logger = setup_logging(name=__file__, log_level='info')
   
    # Load mappings
    logger.info('Loading mappings.')
    news_id2index = load_cache(FILENAME_NEWS_ID2INDEX)
    entity_id2index = load_cache(FILENAME_ENTITY_ID2INDEX)

    # Load subgraph
    logger.info('Loading subgraph.')
    subgraph = load_subgaph()

    # Load articles
    logger.info('Loading articles.')
    articles = load_articles()

    # Retrieve entities for all articles
    logger.info('Retrieving entities per article.')
    news2entities = get_entities_per_article(news_id2index, entity_id2index, subgraph)

    logger.info('Preprocesing training data')
    # Load input file
    data = np.load(FILENAME_TRAIN_RATINGS)
    create_raw_data_file(FILENAME_RAW_TRAIN_DATA)

    logger.info('Preprocesing test data')
    # Load input file
    data = np.load(FILENAME_TEST_RATINGS)
    create_raw_data_file(FILENAME_RAW_TEST_DATA)

    logger.info('Preprocesing random test data')
    # Load input file
    data = np.load(FILENAME_TEST_RATINGS_RANDOM)
    create_raw_data_file(FILENAME_RAW_RANDOM_TEST_DATA)


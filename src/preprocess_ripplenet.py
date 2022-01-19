# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/tezignlab/RippleNet-TF2/blob/master/preprocess.py, which is under an MIT license.

""" Functions for creating the ratings file. """

# import libraries
import numpy as np

# import custom code
from src.config import FILENAME_KG, FILENAME_RATINGS_FINAL_TXT, FILENAME_KG_FINAL_TXT, FILENAME_ENTITY_ID2INDEX, FILENAME_RELATION_ID2INDEX, FILENAME_NEWS_ID2INDEX 
from src.config import DataFrameColumns
from src.fetch_data import get_raw_articles, get_user_item_matrix, get_url2news_node_map
from src.util.logger import setup_logging
from src.util.caching import create_cache, load_cache


def map_news_id2index() -> None:
    """ 
    Maps news node IDs from the knowledge graph to the indices in the news dataframe. 
    The mapping of type :obj:`Dict[str, int]` is cached to disk as a pickle file.
    """
    # Load articles
    articles = get_raw_articles()

    # Load url2news_node map
    url2news_node_map = get_url2news_node_map()

    for idx in range(len(articles)):
        url = articles.iloc[idx][DataFrameColumns.provenance]
        # Not all URLs are identical between the articles dataframe and the KG nodes, due to some unwanted characters at the end of the URL.
        url = clean_url(url)
        indices_in_map = [i for i, key in enumerate(url2news_node_map.keys()) if url in key]

        if indices_in_map:
            key = list(url2news_node_map.keys())[indices_in_map[0]]
            news_node = url2news_node_map[key]
            news_id2index[idx] = news_node
            entity_id2index[news_node] = idx

def convert_rating() -> None:
    """ Creates a file with ratings of type for each user and item from the user-item matrix. """
    logger.info('Reading user-item matrix.')

    # Load user-item matrix
    user_item_matrix = get_user_item_matrix()
    user_item_matrix_np = user_item_matrix.to_numpy()

    item_set = set(news_id2index.keys())
    user_pos_ratings = dict()
   
    for user_idx in range(user_item_matrix_np.shape[0]):
        rated_news_indices = np.where(user_item_matrix_np[user_idx]==1)[0]
        if user_idx not in user_pos_ratings:
            user_pos_ratings[user_idx] = set()
        user_pos_ratings[user_idx].update(list(rated_news_indices))

    logger.info('Converting rating file.')
    writer = open(FILENAME_RATINGS_FINAL_TXT, 'w', encoding='utf-8')
    user_cnt = 0

    for user_idx, pos_item_set in user_pos_ratings.items():
        user_cnt += 1
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_idx, item))

        # There are no negative ratings, only unread news
        unread_set = item_set - pos_item_set
        for item in np.random.choice(list(unread_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_idx, item))

    writer.close()
    logger.info(f'Number of users: {user_cnt}.')
    logger.info(f'Number of items: {len(item_set)}.\n')


def convert_kg() -> None:
    """ Maps entities and relations IDs to indices and creates a file with triples. """
    logger.info('Converting KG file.')

    # Load triples
    triples = load_cache(FILENAME_KG)

    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open(FILENAME_KG_FINAL_TXT, 'w', encoding='utf-8')

    for triple in triples:
        head_old = triple[0]
        relation_old = triple[1]
        tail_old = triple[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    logger.info(f'Number of entities (including articles): {entity_cnt}.')
    logger.info(f'Number of relations: {relation_cnt}.\n')


def clean_url(url: str) -> str:
    """ 
    Removes unwanted characters from the end of certain URLs. 
    
    Args:
        url (:obj:`str`):
            A URL.
    
    Returns:
        :obj:`str`:
            A URL without unwanted characters at the end.
    """
    if 'achgut' in url and '/P' in url:
        return url.rsplit('/P', 1)[0]
    if 'focus' in url:
        return url.rsplit('_id_', 1)[0]
    if 'sueddeutsche' in url and '1.508' in url:
        return url.rsplit('-', 1)[0]
    if 'opposition24' in url:
        return url.rsplit('/?', 1)[0]
    return url


if __name__ == "__main__":
    # Define logger
    logger = setup_logging(name=__file__, log_level='info')
    logger.info('Processing data for RippleNet.\n')

    entity_id2index = dict()
    relation_id2index = dict()
    news_id2index = dict()

    # Map news nodes to article IDs
    map_news_id2index()

    # Convert rating
    convert_rating()

    # Convert knowledge graph
    convert_kg()

    # Cache the mappings
    create_cache(entity_id2index, FILENAME_ENTITY_ID2INDEX)
    create_cache(relation_id2index, FILENAME_RELATION_ID2INDEX)
    create_cache(news_id2index, FILENAME_NEWS_ID2INDEX)
    
    logger.info('Finished.')

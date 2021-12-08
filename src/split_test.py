# -*- coding: utf-8 -*-

""" Function for splitting the test set based on the type of user rating. """

# import libraries
import numpy as np
import pandas as pd
from collections import defaultdict


# import custom code
from src.util.logger import setup_logging
from src.config import FILENAME_USER_RECOMMENDER_MAPPING, FILENAME_TEST_RATINGS, FILENAME_TEST_RATINGS_RANDOM, FILENAME_TEST_RATINGS_NO_TFIDF, FILENAME_TEST_RATINGS_NO_WORD2VEC, FILENAME_TEST_RATINGS_NO_TRANSFORMER


def split_test() -> None:
    """ 
    Splits the complete test set into four smaller test sets of type :obj:`np.ndarray` based on the provenance of user ratings.
    It creates one test set using only ratings based on random recommendations.
    The remaining three test sets are created by removing the ratings based on one recommender, at a time.

    The new test sets are cached to disk as pickle files.
    """

    # Load user-recommender mapping
    user_recommender_mapping = pd.read_csv(FILENAME_USER_RECOMMENDER_MAPPING)
    
    # Map recommender names to IDs 
    recommender_versions = list(zip(user_recommender_mapping['RSversion'].unique(), user_recommender_mapping['RSversion_verbose'].unique()))
    recommender_mapping = defaultdict(list)
    for rec in recommender_versions:
        if 'Random' in rec[1]:
            recommender_mapping['random'].append(rec[0])
        elif 'Transformer' in rec[1]:
            recommender_mapping['transformer'].append(rec[0])
        elif 'Word2vec' in rec[1]:
            recommender_mapping['word2vec'].append(rec[0])
        else:
            recommender_mapping['tfidf'].append(rec[0])

    # Map provenance of rating to users
    rec2user = defaultdict(list)
    for key in recommender_mapping.keys():
        for val in recommender_mapping[key]:
            rec2user[key].extend(list(user_recommender_mapping[user_recommender_mapping['RSversion']==val]['index']))
        logger.info(f'Users with {key}-generated ratings: {len(rec2user[key])}.')

    # Load general test set
    test_ratings = np.load(FILENAME_TEST_RATINGS)

    # Create and cache each test subset
    test_ratings_random = test_ratings[np.isin(test_ratings[:, 0], rec2user['random'])]
    logger.info(f'Random ratings: {test_ratings_random.shape[0]}.')
    np.save(FILENAME_TEST_RATINGS_RANDOM, test_ratings_random)

    test_ratings_no_tfidf = test_ratings[~np.isin(test_ratings[:, 0], rec2user['tfidf'])]
    logger.info(f'Ratings w/o TFIDF: {test_ratings_no_tfidf.shape[0]}.')
    np.save(FILENAME_TEST_RATINGS_NO_TFIDF, test_ratings_no_tfidf)
    
    test_ratings_no_word2vec = test_ratings[~np.isin(test_ratings[:, 0], rec2user['word2vec'])]
    logger.info(f'Ratings w/o Word2vec: {test_ratings_no_word2vec.shape[0]}.')
    np.save(FILENAME_TEST_RATINGS_NO_WORD2VEC, test_ratings_no_word2vec)

    test_ratings_no_transformer = test_ratings[~np.isin(test_ratings[:, 0], rec2user['transformer'])]
    logger.info(f'Ratings w/o Transformer: {test_ratings_no_transformer.shape[0]}.')
    np.save(FILENAME_TEST_RATINGS_NO_TRANSFORMER, test_ratings_no_transformer)


if __name__ == '__main__':
    logger = setup_logging(name=__file__, log_level='info')

    logger.info('Splitting test data based on rating provenance.')
    split_test()
    logger.info('Finished.')

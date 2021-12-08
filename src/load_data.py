# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/tezignlab/RippleNet-TF2/blob/master/tools/load_data.py, which is under an MIT license.

""" Utilities for data loading for RippleNet. """

# import libraries
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# import custom code
from src.util.logger import setup_logging
from src.util.caching import create_cache, load_cache
from src.config import FILENAME_RATINGS_FINAL_TXT, FILENAME_RATINGS_FINAL_NPY, FILENAME_KG_FINAL_TXT, FILENAME_KG_FINAL_NPY, FILENAME_TRAIN_RATINGS, FILENAME_USER_HISTORY_DICT
from src.config import FILENAME_TEST_RATINGS, FILENAME_TEST_RATINGS_RANDOM, FILENAME_TEST_RATINGS_NO_TFIDF, FILENAME_TEST_RATINGS_NO_WORD2VEC, FILENAME_TEST_RATINGS_NO_TRANSFORMER


class LoadData:
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(name=__file__, log_level='info')

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, int, int, Dict[int, List[Tuple[int, int, int]]]]:
        """ 
        Loads and returns the data needed in RippleNet.

        Returns:
            - :obj:`np.ndarray`: 
                Training set of ratings.
            - :obj:`np.ndarray`:
                Test set of ratings.
            - :obj:`int`:
                Number of entities.
            - :obj:`int`:
                Number of relations.
            - :obj:`Dict[int, List[Tuple[int, int, int]]]`:
                Ripple sets of each user.
        """

        train_data, test_data, user_history_dict = self.load_rating()
        n_entity, n_relation, kg = self.load_kg()
        ripple_set = self.get_ripple_set(kg, user_history_dict)
        return train_data, test_data, n_entity, n_relation, ripple_set

    def get_test_file(self, test_set_type: str) -> Path:
        """ 
        Retrieves the filepath of a test set given its type.

        Args:
            test_set_type (:obj:`str`):
                The type of test set.

        Returns:
            :obj:`Path`:
                The filepath of the test set.
        """
        test_set_type2file = {
            'complete': FILENAME_TEST_RATINGS,
            'random': FILENAME_TEST_RATINGS_RANDOM,
            'no_tfidf_ratings': FILENAME_TEST_RATINGS_NO_TFIDF,
            'no_word2vec_ratings': FILENAME_TEST_RATINGS_NO_WORD2VEC,
            'no_transformer_ratings': FILENAME_TEST_RATINGS_NO_TRANSFORMER
        }
        return test_set_type2file[test_set_type]

    def load_rating(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
        """
        It loads the training and test data, and the user history, if they exist.
        Otherwise, it loads the user ratings, processes them to construct the training and test sets, and user history, and caches them to disk.

        Returns:
            - :obj:`np.ndarray`: 
                Training set of ratings.
            - :obj:`np.ndarray`: 
                Test set of ratings.
            - :obj:`Dict[int, List[int]]`: 
                User history dictionary.
        """
        self.logger.info('Reading rating file.')

        test_file = self.get_test_file(self.args.test_set)        

        if os.path.exists(FILENAME_TRAIN_RATINGS) and os.path.exists(test_file) and os.path.exists(FILENAME_USER_HISTORY_DICT):
            self.logger.info('Loading training and test data.')
            train_data = np.load(FILENAME_TRAIN_RATINGS)
            test_data = np.load(test_file)
            user_history_dict = load_cache(FILENAME_USER_HISTORY_DICT)

            self.logger.info(f'Size training data: {train_data.shape}.')
            self.logger.info(f'Size test data: {test_data.shape}.')
        else:
            # Read rating file
            if os.path.exists(FILENAME_RATINGS_FINAL_NPY):
                rating_np = np.load(FILENAME_RATINGS_FINAL_NPY)
            else:
                rating_np = np.loadtxt(FILENAME_RATINGS_FINAL_TXT, dtype=np.int32)
                np.save(FILENAME_RATINGS_FINAL_NPY, rating_np)

            # Split dataset
            self.logger.info('Splitting dataset.')
            test_ratio = 0.2
            n_ratings = rating_np.shape[0]

            test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
            train_indices = set(range(n_ratings)) - set(test_indices)

            # Traverse training data, only keeping the users with positive ratings
            user_history_dict = dict()
            for i in train_indices:
                user = rating_np[i][0]
                item = rating_np[i][1]
                rating = rating_np[i][2]
                if rating == 1:
                    if user not in user_history_dict:
                        user_history_dict[user] = []
                    user_history_dict[user].append(item)

            train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
            test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

            train_data = rating_np[train_indices]
            test_data = rating_np[test_indices]
            self.logger.info(f'Size training data: {train_data.shape}.')
            self.logger.info(f'Size test data: {test_data.shape}.')

            # Cache test and train data
            np.save(FILENAME_TRAIN_RATINGS, train_data)
            np.save(FILENAME_TEST_RATINGS, test_data)
            create_cache(user_history_dict, FILENAME_USER_HISTORY_DICT)

            self.logger.info('Finished.\n')

        return train_data, test_data, user_history_dict

    def load_kg(self) -> Tuple[int, int, Dict[int, List[Tuple[int, int]]]]:
        """ 
        Loads the knowledge graph if already cached as :obj:`np.ndarray`, otherwise it constructs it from the text file. 
        
        Returns:
            - :obj:`int`: 
                Number of entities.
            - :obj:`int`:
                Number of relations.
            - :obj:`Dict[int, List[Tuple[int, int]]]`:
                The knowledge graph as a dictionary which maps each head entity to a tuple of the form (tail, relation).
        """
        self.logger.info('Reading KG file.')

        # Reading KG file
        if os.path.exists(FILENAME_KG_FINAL_NPY):
            kg_np = np.load(FILENAME_KG_FINAL_NPY)
        else:
            kg_np = np.loadtxt(FILENAME_KG_FINAL_TXT, dtype=np.int32)
            np.save(FILENAME_KG_FINAL_NPY, kg_np)

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1]))

        self.logger.info('Constructing knowledge graph.')
        kg = defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((tail, relation))

        self.logger.info('Finished.\n')

        return n_entity, n_relation, kg

    def get_ripple_set(self, kg: Dict[int, List[Tuple[int, int]]], user_history_dict: Dict[int, List[int]]) -> Dict[int, List[Tuple[int, int, int]]]:
        """ 
        Creates the ripple set for each user. 

        Args:
            kg (:obj:`Dict[int, List[Tuple[int, int]]]`):
                The knowledge graph as a dictionary which maps each head entity to a tuple of the form (tail, relation).
            user_history_dict (:obj:`Dict[int, List[int]]`):
                User history dictionary.

        Returns:
            :obj:`Dict[int, List[Tuple[int, int, int]]]`:
                Ripple sets of each user.
        """
        self.logger.info('Constructing ripple set.')

        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        ripple_set = defaultdict(list)

        for user in user_history_dict:
            for h in range(self.args.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                """
                If the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                This won't happen for h = 0, because only the items that appear in the KG have been selected.
                """
                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # Sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.args.n_memory
                    indices = np.random.choice(len(memories_h), size=self.args.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))
        
        self.logger.info('Finished.\n')
        return ripple_set

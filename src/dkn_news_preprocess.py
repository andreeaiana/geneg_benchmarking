# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted fromi https://github.com/hwwang55/DKN/blob/master/data/news/news_preprocess.py

# import libraries
import re
import os
import gensim
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# import custom code
from src.config import FILENAME_RAW_TRAIN_DATA, FILENAME_RAW_TEST_DATA, FILENAME_TRAIN_DATA, FILENAME_TEST_DATA, FILENAME_RAW_RANDOM_TEST_DATA, FILENAME_RANDOM_TEST_DATA, FILENAME_ENTITY2INDEX
from src.config import MODELS_DIR, DATA_DIR
from src.config import MAX_ARTICLE_LENGTH, W2V_WORD_EMBEDDING_DIM
from src.util.logger import setup_logging


PATTERN1 = re.compile('[^A-Za-z]')
PATTERN2 = re.compile('[ ]{2,}')
WORD_FREQ_THRESHOLD = 2
ENTITY_FREQ_THRESHOLD = 1

word2freq = {}
entity2freq = {}
word2index = {}
entity2index = {}
corpus = []


def count_word_and_entity_freq(files: List[Path]) -> None:
    """
    Counts the frequency of words and entities in news titles in the training and test files.

    Args:
        files (:obj:`List[Path]`): 
            [training_file, test_file]
    """
    for file in files:
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.strip().split('\t')
            news_title = array[1]
            entities = array[3]

            # count word frequency
            for s in news_title.split(' '):
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity frequency
            for s in entities.split(';'):
                entity_id = s[:s.index(':')]
                if entity_id not in entity2freq:
                    entity2freq[entity_id] = 1
                else:
                    entity2freq[entity_id] += 1

            corpus.append(news_title.split(' '))
        reader.close()


def construct_word2id_and_entity2id() -> None:
    """
    Allocates each valid word and entity a unique index (start from 1).
    """
    cnt = 1  # 0 is for dummy word
    for w, freq in word2freq.items():
        if freq >= WORD_FREQ_THRESHOLD:
            word2index[w] = cnt
            cnt += 1
    logger.info('- word size: %d' % len(word2index))

    writer = open(FILENAME_ENTITY2INDEX, 'w', encoding='utf-8')
    cnt = 1
    for entity, freq in entity2freq.items():
        if freq >= ENTITY_FREQ_THRESHOLD:
            entity2index[entity] = cnt
            writer.write('%s\t%d\n' % (entity, cnt))  # for later use
            cnt += 1
    writer.close()
    logger.info('- entity size: %d' % len(entity2index))


def get_local_word2entity(entities: str) -> Dict[str, int]:
    """
    Given the entities information in one line of the dataset, construct a map from word to entity index.

    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
    'potter':index_of(id_1), 'england': index_of(id_2)}

    Args:
        entities (:obj:`str`):
            Entities information in one line of the dataset.

    Returns:
        :obj:`Dict[str, int]`:
            A local map from word to entity index.
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        entity_name = entity_pair[entity_pair.index(':') + 1:]

        # remove non-character word and transform words to lower case
        entity_name = PATTERN1.sub(' ', entity_name)
        entity_name = PATTERN2.sub(' ', entity_name).lower()

        # constructing map: word -> entity_index
        for w in entity_name.split(' '):
            entity_index = entity2index[entity_id]
            local_map[w] = entity_index

    return local_map


def encoding_title(title: str, entities: str) -> Tuple[str, str]:
    """
    Encoding a title according to word2index map and entity2index map.

    Args:
        title (:obj:`str`):
            A piece of news title
        entities (:obj:`str`):
            Entities contained in the news title
    Returns:
        - :obj:`str`:
            Encodings of the title with respect to word 
        - :obj:`str`:
            Encodings of the title with respect to entity
    """
    local_map = get_local_word2entity(entities)

    array = title.split(' ')
    word_encoding = ['0'] * MAX_ARTICLE_LENGTH
    entity_encoding = ['0'] * MAX_ARTICLE_LENGTH

    point = 0
    for s in array:
        if s in word2index:
            word_encoding[point] = str(word2index[s])
            if s in local_map:
                entity_encoding[point] = str(local_map[s])
            point += 1
        if point == MAX_ARTICLE_LENGTH:
            break
    word_encoding = ','.join(word_encoding)
    entity_encoding = ','.join(entity_encoding)
    return word_encoding, entity_encoding


def transform(input_file: Path, output_file: Path) -> None:
    reader = open(input_file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        user_id = array[0]
        title = array[1]
        label = array[2]
        entities = array[3]
        word_encoding, entity_encoding = encoding_title(title, entities)
        writer.write('%s\t%s\t%s\t%s\n' % (user_id, word_encoding, entity_encoding, label))
    reader.close()
    writer.close()


def get_word2vec_model() -> gensim.models.Word2Vec:
    model_filepath = os.path.join(MODELS_DIR, 'word_embeddings_' + str(W2V_WORD_EMBEDDING_DIM) + '.model')
    if not os.path.exists(model_filepath):
        logger.info('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, vector_size=W2V_WORD_EMBEDDING_DIM, min_count=1, workers=8)
        logger.info('- saving model ...')
        w2v_model.save(model_filepath)
    else:
        logger.info('- loading model ...')
        w2v_model = gensim.models.word2vec.Word2Vec.load(model_filepath)
    return w2v_model


if __name__ == '__main__':
    logger = setup_logging(name=__file__, log_level='info')

    logger.info('counting frequencies of words and entities ...')
    count_word_and_entity_freq([FILENAME_RAW_TRAIN_DATA, FILENAME_RAW_TEST_DATA])

    logger.info('constructing word2id map and entity to id map ...')
    construct_word2id_and_entity2id()

    logger.info('transforming training and test dataset ...')
    transform(FILENAME_RAW_TRAIN_DATA, FILENAME_TRAIN_DATA)
    transform(FILENAME_RAW_TEST_DATA, FILENAME_TEST_DATA)
    transform(FILENAME_RAW_RANDOM_TEST_DATA, FILENAME_RANDOM_TEST_DATA)

    logger.info('getting word embeddings ...')
    embeddings = np.zeros([len(word2index) + 1, W2V_WORD_EMBEDDING_DIM])
    model = get_word2vec_model()
    for index, word in enumerate(word2index.keys()):
        embedding = model.wv[word] if word in model.wv.key_to_index else np.zeros(W2V_WORD_EMBEDDING_DIM)
        embeddings[index + 1] = embedding
    logger.info('- writing word embeddings ...')
    np.save(os.path.join(DATA_DIR, 'word_embeddings_' + str(W2V_WORD_EMBEDDING_DIM)), embeddings)

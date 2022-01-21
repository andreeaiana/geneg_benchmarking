# -*- coding: utf-8 -*-

# this file contains all variables that are set by the user and other configuration options

# import libraries
import nltk
import shutil
from pathlib import Path
from dataclasses import dataclass
from fasttext.util import download_model


# Define directories
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Model names
BASE_NAME_FASTTEXT = 'cc.de.300.bin'
BASE_NAME_RIPPLENET = 'ripple_net_model.h5'
BASE_NAME_TRANSE = 'transe.ckpt'
BASE_NAME_TRANSD = 'transd.ckpt'
BASE_NAME_TRANSH = 'transh.ckpt'
BASE_NAME_TRANSR = 'transr.ckpt'
BASE_NAME_TRANSR_TRANSE = 'transr_transe.json'

# Parameters
K_LIST = [1, 2, 5, 10, 20, 50, 100] # Positions to evaluate top-k recommendation on
DKN_KGE_METHOD = "TransE"
DKN_KGE_ENTITY_EMBEDDING_DIM = 50
MAX_ARTICLE_LENGTH = 400
W2V_WORD_EMBEDDING_DIM = 200


# Define filenames
FILENAME_ARTICLES = DATA_DIR / "articles.csv"
FILENAME_USER_ITEM_MATRIX = DATA_DIR / "user_item_matrix.csv"
FILENAME_URL2NEWS_NODE_MAP = DATA_DIR / "url2news_node_map.p"
FILENAME_KG = DATA_DIR / "kg.p"
FILENAME_SUBGRAPH = DATA_DIR / "subgraph.p"
FILENAME_KG_FINAL_TXT = DATA_DIR / "kg_final.txt"
FILENAME_KG_FINAL_NPY = DATA_DIR / "kg_final.npy"
FILENAME_RATINGS_FINAL_TXT = DATA_DIR / "ratings_final.txt"
FILENAME_RATINGS_FINAL_NPY = DATA_DIR / "ratings_final.npy"
FILENAME_USER_HISTORY_DICT = DATA_DIR / 'user_history_dict.p'
FILENAME_TRAIN_RATINGS = DATA_DIR / "train_ratings.npy"
FILENAME_TEST_RATINGS = DATA_DIR / "test_ratings.npy"
FILENAME_TEST_RATINGS_RANDOM = DATA_DIR / 'test_ratings_random.npy'
FILENAME_TEST_RATINGS_NO_TFIDF = DATA_DIR / 'test_ratings_no_tfidf.npy'
FILENAME_TEST_RATINGS_NO_WORD2VEC = DATA_DIR / 'test_ratings_no_word2vec.npy'
FILENAME_TEST_RATINGS_NO_TRANSFORMER = DATA_DIR / 'test_ratings_no_transformer.npy'
FILENAME_ENTITY_ID2INDEX = DATA_DIR / "entity_id2index.p"
FILENAME_RELATION_ID2INDEX = DATA_DIR/ "relation_id2index.p"
FILENAME_NEWS_ID2INDEX = DATA_DIR / "news_id2index.p"
FILENAME_USER_RECOMMENDER_MAPPING = DATA_DIR / 'user_recommender_mapping.csv'
FILENAME_RAW_TRAIN_DATA = DATA_DIR / "raw_train.txt"
FILENAME_TRAIN_DATA = DATA_DIR / "train.txt"
FILENAME_RAW_TEST_DATA = DATA_DIR / "raw_test.txt"
FILENAME_RAW_RANDOM_TEST_DATA = DATA_DIR / "raw_random_test.txt"
FILENAME_TEST_DATA = DATA_DIR / "test.txt"
FILENAME_RANDOM_TEST_DATA = DATA_DIR / "random_test.txt"
FILENAME_ENTITY2INDEX = DATA_DIR / "entity2index.txt"
FILENAME_ENTITY2ID = DATA_DIR / "entity2id.txt"
FILENAME_RELATION2ID = DATA_DIR/ "relation2id.txt"
FILENAME_TRIPLE2ID = DATA_DIR/ "train2id.txt"

FILENAME_FASTTEXT_MODEL = MODELS_DIR / BASE_NAME_FASTTEXT
FILENAME_RIPPLENET_MODEL = MODELS_DIR / BASE_NAME_RIPPLENET
FILENAME_TRANSE_MODEL = MODELS_DIR / BASE_NAME_TRANSE
FILENAME_TRANSD_MODEL = MODELS_DIR / BASE_NAME_TRANSD
FILENAME_TRANSH_MODEL = MODELS_DIR / BASE_NAME_TRANSH
FILENAME_TRANSR_MODEL = MODELS_DIR / BASE_NAME_TRANSR
FILENAME_TRANSR_TRANSE_MODEL = MODELS_DIR / BASE_NAME_TRANSR_TRANSE


# Create directories if they do not exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    if directory.exists() == False:
        directory.mkdir()

# Class that corresponds to the column names in dataframe
@dataclass        
class DataFrameColumns:
    author = 'author_person'
    title = 'title'
    content = 'content'
    sentiment = 'sentiment_score'
    provenance = 'provenance'
    preprocessed_content = 'preprocessed_content'
    
# Variables used for replacing certain expressions
PLACEHOLDER_OUTLET = 'Medium'

# Get stopwords from external sources
try:
    # Assign variable
    STOP_WORDS = set(nltk.corpus.stopwords.words('german'))

except Exception:

    # Download nltk vocab
    nltk.download('stopwords')
    STOP_WORDS = set(nltk.corpus.stopwords.words('german'))
    
# Download fasttext model
if not FILENAME_FASTTEXT_MODEL.exists():
    
    source_filename = Path.cwd() / BASE_NAME_FASTTEXT
    download_model('de')
    shutil.move(source_filename, FILENAME_FASTTEXT_MODEL)
    

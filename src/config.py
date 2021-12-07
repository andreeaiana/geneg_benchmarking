# this file contains all variables that are set by the user and other configuration options

# import libraries
import nltk
import shutil
from pathlib import Path
from dataclasses import dataclass
from fasttext.util import download_model

# define directories
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# var names
BASE_NAME_FASTTEXT = 'cc.de.300.bin'
BASE_NAME_RIPPLENET = 'ripple_net_model.h5'

# parameters
K_LIST = [1, 2, 5, 10, 20, 50, 100] # Positions to evaluate top-k recommendation on

# define filenames
FILENAME_ARTICLES = DATA_DIR / "articles.csv"
FILENAME_USER_ITEM_MATRIX = DATA_DIR / "user_item_matrix.csv"
FILENAME_URL2NEWS_NODE_MAP = DATA_DIR / "url2news_node_map.p"
FILENAME_KG = DATA_DIR / "kg.p"
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
FIELNAME_ENTITY_ID2INDEX = DATA_DIR / "entity_id2index.p"
FILENAME_RELATION_ID2INDEX = DATA_DIR/ "relation_id2index.p"
FILENAME_NEWS_ID2INDEX = DATA_DIR / "news_id2index.p"
FILENAME_USER_RECOMMENDER_MAPPING = DATA_DIR / 'user_recommender_mapping.csv'

FILENAME_FASTTEXT_MODEL = MODELS_DIR / BASE_NAME_FASTTEXT
FILENAME_RIPPLENET_MODEL = MODELS_DIR / BASE_NAME_RIPPLENET

# create directories if they do not exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    if directory.exists() == False:
        directory.mkdir()

# class that corresponds to the column names in dataframe
@dataclass        
class DataFrameColumns:
    author = 'author_person'
    title = 'title'
    content = 'content'
    sentiment = 'sentiment_score'
    provenance = 'provenance'
    
# variables used for replacing certain expressions
PLACEHOLDER_OUTLET = 'Medium'

# get stopwords from external sources
try:
    # assign variable
    STOP_WORDS = set(nltk.corpus.stopwords.words('german'))

except Exception:

    # download nltk vocab
    nltk.download('stopwords')
    STOP_WORDS = set(nltk.corpus.stopwords.words('german'))
    
# download fasttext model
if not FILENAME_FASTTEXT_MODEL.exists():
    
    source_filename = Path.cwd() / BASE_NAME_FASTTEXT
    download_model('de')
    shutil.move(source_filename, FILENAME_FASTTEXT_MODEL)
    

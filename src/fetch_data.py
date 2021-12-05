# this file contains all the functions related to data retrieval 

# import libraries
import pickle
import pandas as pd
import numpy as np

# import custom code
from src.config import FILENAME_ARTICLES, FILENAME_USER_ITEM_MATRIX, FILENAME_URL2NEWS_NODE_MAP

def get_raw_articles() -> pd.DataFrame:
    return pd.read_csv(FILENAME_ARTICLES)

def get_user_item_matrix() -> pd.DataFrame:
    return pd.read_csv(FILENAME_USER_ITEM_MATRIX, sep=';').drop('id', axis=1)

def get_url2news_node_map() -> dict:
    with open(FILENAME_URL2NEWS_NODE_MAP, 'rb') as f:
        user2news_node_map = pickle.load(f)
    return user2news_node_map

def get_item_history_from_user(user_id: int, data: np.ndarray) -> list:
    
        # get mask for articles that user has read in the past
        mask = np.logical_and(data[:,0] == user_id, data[:,2] == 1)
        items = list(data[mask,1])
        
        return items

if __name__ == '__main__':
    
    # get data
    articles = get_raw_articles()
    user_item_matrix = get_user_item_matrix()
    user2news_node_map = get_url2news_node_map()
    
    # print output
    print(articles)
    print(user_item_matrix)
    print(user2news_node_map)

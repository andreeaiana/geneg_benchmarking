# import libraries
import pandas as pd
import argparse
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# import custom Code
from src.config import FILENAME_TRAIN_RATINGS, MODELS_DIR, DATA_DIR
from src.models import MODELS
from src.metrics import auc, topk_eval
from src.util.logger import setup_logging
from src.fetch_data import get_item_history_from_user

def normalize_scores_with_filter(scores, items2exclude):
    
    scores_filtered = np.delete(scores, items2exclude)
    min_value, max_value = scores_filtered.min(), scores_filtered.max()
    scores_normalized = (scores - min_value) / (max_value - min_value)
    
    return scores_normalized

def predict_wrapper(model, row):
    
    # extract information
    user = row[0]
    item = row[1]
    
    # get item history of user
    user_item_history = get_item_history_from_user(user_id=user, data=X_train)
    
    if model.__class__.__base__.__name__ == 'TemplateContentRecommender':
        
        # similarity scores for already seen items
        sim_scores = model.sim_matrix_[user_item_history,:]
        
        # calculate mean similarity scores
        sim_scores_mean = sim_scores.mean(axis=0)
        
        # normalize similarity scores
        # exclude previously seen articles from calculating the min and max values for normalizing
        # this ensure that the remaining similarity scores range from 0 to 1
        sim_scores_mean_normalized = normalize_scores_with_filter(scores=sim_scores_mean, items2exclude=user_item_history)   
        
        # retrieve corresponding item score
        item_score = sim_scores_mean_normalized[item]
        #item_score = sim_scores_mean[item]
        
    else:
        
        user_items = model.item_user_matrix.T.tocsr()
        score_explainer_list = [model.model.explain(user, user_items, item) for item in range(model.item_user_matrix.shape[0])]
        scores = np.array([el[0] for el in score_explainer_list])
        scores_normalized = normalize_scores_with_filter(scores=scores, items2exclude=user_item_history) 
        item_score = scores_normalized[item]
    
    # predict item score       
    return item_score
    

def evaluate(name):

    # load model 
    logger.info(f'Loading {name} model')
    filename = MODELS_DIR / (name + '.p')
    
    with open(filename, 'rb') as file:
        model = pickle.load(file=file)
    
    # get predictions
    y_pred_scores = [predict_wrapper(model, row) for row in tqdm(X_test)]
    y_pred = [int(score > 0.5) for score in y_pred_scores]
    
    # calculate score
    accuracy_val = accuracy_score(y_true=X_test[:,2], y_pred=y_pred)    
    auc_val = auc(X_test[:, 2].astype('float32'), y_pred_scores)
    f1_val = f1_score(y_true=X_test[:,2], y_pred=y_pred)
    precision_val = precision_score(y_true=X_test[:,2], y_pred=y_pred)
    recall_val = recall_score(y_true=X_test[:,2], y_pred=y_pred)
    
    # log results
    logger.info(f"Accuracy = {accuracy_val}")
    logger.info(f"Auc = {auc_val}")
    logger.info(f"F1 = {f1_val}")
    logger.info(f"Precision = {precision_val}")
    logger.info(f"Recall = {recall_val}")
    logger.info(f"{accuracy_val} {auc_val} {precision_val} {recall_val} {f1_val}")

if __name__ == '__main__':
    
    # define logger
    logger = setup_logging(__file__)
    
    # define parser
    parser = argparse.ArgumentParser(description='train recommender')
    parser.add_argument('-m', '--model', default=None, type=str, help='type of recommender')
    parser.add_argument('-d', '--data', default=None, type=str, help='name of dataset')

    # get args
    args = parser.parse_args()
    model_name = args.model
    data_set = args.data
    
    # get data of shape (n_observations, 3)), whereby 
    # the first column represents the user id, 
    # the second column represents the item id and 
    # the third column represents the rating
    X_train =  np.load(str(FILENAME_TRAIN_RATINGS))
    
    # evaluate on different test sets
    for filename in DATA_DIR.glob('test_*.npy'):
        
        if data_set not in str(filename):
            continue
        
        # get test set
        X_test = np.load(str(filename))
        
        # evaluate model
        print('\n')
        logger.info(f'Using test set: {str(filename)}')    
        logger.info(f'Baseline for accuracy of classifier: {X_test[:,2].sum()/X_test.shape[0]}')
        
        # if explicit model selected evaluate only on this model
        if model_name:
            evaluate(model_name)
        else:
        
            # get model name    
            for name, _ in MODELS.items():            
                evaluate(name)
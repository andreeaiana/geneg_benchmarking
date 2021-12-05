# app.py

"""
This python file contains all the models used for ReNewsRs project.
"""

# import libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging
import fasttext
import fasttext.util
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sentence_transformers import SentenceTransformer
from implicit.als import AlternatingLeastSquares

# import custom code
from src.fetch_data import get_item_history_from_user, get_user_item_matrix
from src.preprocess import preprocess
from src.config import MODELS_DIR, FILENAME_FASTTEXT_MODEL
from src.util.logger import setup_logging


class TemplateContentRecommender(BaseEstimator, ClassifierMixin):
    """
    Base class for recommender systems used in this project.

    Template for class taken and adapted from https://scikit-learn.org/stable/developers/develop.html

    """

    def __init__(self):

        # placeholders for text, titles and similarities
        self.X_ = None
        self.y_ = None
        self.labels = None
        self.sim_matrix_ = None
        self.bool_use_history = True
        self.logger = setup_logging(name=Path(__file__).name, log_level='info')

        # preprocessing options
        self._bool_remove_html_tags = True
        self._bool_remove_links = True
        self._bool_remove_special_symbols = True
        self._bool_remove_punctuation = True
        self._bool_seperate_numbers_from_text = True
        self._bool_stemming = True
        self._bool_word_tokenize = True

    def set_labels(self, X: pd.Series):
        try:
            self.labels = X.index.to_list()
        except Exception:
            pass

    def preprocess(self, X: pd.Series) -> pd.Series:

        # result placeholder
        result = []

        for text in X:

            text_preprocessed = preprocess(text=text,
                                           bool_remove_html_tags=self._bool_remove_html_tags,
                                           bool_remove_links=self._bool_remove_links,
                                           bool_remove_special_symbols=self._bool_remove_special_symbols,
                                           bool_remove_punctuation=self._bool_remove_punctuation,
                                           bool_seperate_numbers_from_text=self._bool_seperate_numbers_from_text,
                                           bool_stemming=self._bool_stemming,
                                           bool_word_tokenize=self._bool_word_tokenize,
                                           )
            result.append(text_preprocessed)

        return pd.Series(result, index=X.index)
    
    def single_predict(self, X: list, n=10) -> list:

        # Check is fit had been called
        check_is_fitted(self)

        if self.bool_use_history:
            scores_cb = self.sim_matrix_[X,:].mean(axis=0)
        else:
            # focus only on last element the user has seen
            scores_cb = self.sim_matrix_[X[-1], :]

        # exclude already read article from suggestions
        mask = np.ones((self.sim_matrix_.shape[0],), dtype=bool)
        mask[X] = False

        # get indices of recommended articles
        recommendations = pd.Series(scores_cb)[mask].sort_values(ascending=False).head(n).index.to_list()

        # return recommendations
        return recommendations
    
    def predict(self, X_train: np.ndarray, X_test_user_ids: np.ndarray, n=10) -> list:
        
        # result placeholder for recommendations
        results = []
               
        for user_id in X_test_user_ids:
            
            # get user history
            read_articles = get_item_history_from_user(user_id=user_id, data=X_train)
            
            # single predict
            y_pred = self.single_predict(X=read_articles, n=n)
            
            # add to result
            results.append(y_pred)
        
        return results


class TfidfRecommender(TemplateContentRecommender):

    def __init__(self):

        # init parent class
        TemplateContentRecommender.__init__(self)

    def fit(self, X, y):

        # set labels
        self.labels = y

        # preprocess input
        X_preprocessed = self.preprocess(X)

        # define vectorizer
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), norm="l2")

        # fit vectorizer with input X and transform the input subsequently
        self.X_ = vectorizer.fit_transform(X_preprocessed)

        # calculate cosine similarities between documents
        self.sim_matrix_ = cosine_similarity(self.X_)

        return self
    
    
class FastTextRecommender(TemplateContentRecommender):

    def __init__(self):
        TemplateContentRecommender.__init__(self)

        # overwrite preprocessing attributes from TemplateRecommender
        self._bool_stemming = False

    def fit(self, X, y):

        # set labels
        self.labels = y

        # load fastText model
        self.logger.info('Loading model')
        ft = fasttext.load_model(str(FILENAME_FASTTEXT_MODEL))

        # preprocess input
        X_preprocessed = self.preprocess(X)

        # placeholder for all document vectors
        result = []

        # transform input into numeric vectors representing the document
        self.logger.info('Transforming text into numeric vectors')
        for text in tqdm(X_preprocessed):

            words = text.split()

            # placeholder
            doc_vec = 0

            # get weighted word vector and add to doc vector
            for word in words:
                doc_vec += ft.get_word_vector(word) / len(words)

            result.append(doc_vec)

        # store results
        self.X_ = np.asarray(result)

        # calculate cosine similarities between documents
        self.logger.info('Calculating cosine similarity')
        self.sim_matrix_ = cosine_similarity(self.X_)

        return self


class TransformerRecommender(TemplateContentRecommender):

    def __init__(self):
        TemplateContentRecommender.__init__(self)

        # overwrite preprocessing attributes from TemplateRecommender
        self._bool_word_tokenize = False
        self._bool_remove_punctuation = False

    def fit(self, X, y):

        # set labels
        self.labels = y

        # load Transformer model
        self.logger.info('Loading model')
        model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

        # disable logging
        logging.getLogger().setLevel(logging.WARNING)

        # is first run
        is_first_run = True

        for text in tqdm(X, total=X.shape[0]):

            # get sentences from text
            sentences = self.preprocess(pd.Series(text))

            # get sentence embedding
            embeddings = model.encode(sentences, convert_to_tensor=True)

            # calculate average of sentence embedding
            embeddings_average = embeddings.mean(axis=0).reshape(1, -1)

            if is_first_run:
                result = embeddings_average
                is_first_run = False
            else:
                result = torch.cat((result, embeddings_average), 0)

        # store results
        self.X_ = result.numpy()

        # calculate cosine similarities between documents
        self.logger.info('Calculating cosine similarity')
        self.sim_matrix_ = cosine_similarity(self.X_)


class TemplateCollaborativeRecommender(BaseEstimator, ClassifierMixin):
    
    def __init__(self):

        # placeholders for text, titles and similarities
        self.X_ = None
        self.y_ = None
        self.labels = None
        self.item_user_matrix = None
        self.logger = setup_logging(name=Path(__file__).name, log_level='info')
        self.n_user = get_user_item_matrix().index.nunique()
        self.n_items = get_user_item_matrix().columns.nunique()
        
    def set_labels(self, X: pd.Series):
        try:
            self.labels = X.index.to_list()
        except Exception:
            pass


class ALSModel(TemplateCollaborativeRecommender):
    """
    Wrapper model for from AlternatingLeastSquares from implicit package
    """
    
    def __init__(self, n_factors: int = 50):

        # init parent class
        TemplateCollaborativeRecommender.__init__(self)
        
        # init parameters
        self.n_factors = n_factors
        self.model = AlternatingLeastSquares(factors=n_factors, random_state=0)
        
    def create_item_user_matrix(self, X: np.ndarray) -> csr_matrix:
        """
        Create user-item matrix from input dataframe
        Parameters
        ----------
        X : np.ndarray
            numpy array containing user-item interactions
        Returns
        -------
        item_user_matrix : csr_matrix
            item-user matrix
        """
        
        # transform input
        user_item_matrix = pd.DataFrame(X).pivot(index=[0], columns=[1])
        user_item_matrix = user_item_matrix.droplevel(0,axis=1)
        user_item_matrix.index.name = 'user_id'
        user_item_matrix.columns.name = 'items_id'
        
        # add missing rows
        for user in list(range(self.n_user)):

            if user not in user_item_matrix.index:
                tmp = pd.DataFrame(index=[user], columns=user_item_matrix.columns)
                user_item_matrix = user_item_matrix.append(tmp)
                del tmp
        
        # add missing columns
        for item in list(range(self.n_items)):
            
            if item not in user_item_matrix.columns:
                tmp = pd.DataFrame(index=user_item_matrix.index, columns=[item])
                user_item_matrix = pd.concat([user_item_matrix, tmp], axis=1)
                del tmp
                
        # sort indices and columns       
        user_item_matrix.sort_index(inplace=True)
        user_item_matrix.sort_index(axis=1, inplace=True)
        
        # final sanity check
        assert user_item_matrix.shape == (self.n_user, self.n_items)
        
        item_user_matrix = csr_matrix(user_item_matrix.fillna(0).T.values)

        # return user-item matrix
        return item_user_matrix
        
    def fit(self, X, y):

        # set labels
        self.labels = y
        
        # transform X to item_user_matrix
        self.item_user_matrix = self.create_item_user_matrix(X)

        # fit model
        self.model.fit(self.item_user_matrix)
        
        return self
    
    def single_predict(self, user_id: int, n: int=10) -> list:

        # Check is fit had been called
        check_is_fitted(self)
        
        # infer user item matrix
        user_items = self.item_user_matrix.T.tocsr()

        # get indices of recommended articles
        recommendations = [el[0] for el in self.model.recommend(user_id, user_items, N=n)]

        # return recommendations
        return recommendations
    
    def predict(self, X_test_user_ids: np.ndarray, n=10) -> list:
        
        # result placeholder for recommendations
        results = []
        
        # loop through all users       
        for user_id in X_test_user_ids:
            
            # single predict
            y_pred = self.single_predict(user_id=user_id, n=n)
            
            # add to result
            results.append(y_pred)
        
        return results
    
# provide summary of all models
MODELS = {
    'tfidf': TfidfRecommender(),
    'word2vec': FastTextRecommender(),
    'transformer': TransformerRecommender(),
    'als': ALSModel()
}
# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/tezignlab/RippleNet-TF2/blob/master/tools/metrics.py, which is under an MIT license.

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score

# import custom code
from src.fetch_data import get_item_history_from_user


def auc(y_true, y_pred):
    # cal auc
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP / P


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def topk_eval(model, train_data, test_data, k_list):
    
    # Adapted from https://github.com/hwwang55/MKR/blob/master/src/train.py
    
    # placeholder for results
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    
    # infer user_data from test data
    user_list = test_data[:,0]
    
    # calculate hits for each user individually
    for user in user_list:
        
        # retrieve items that user has seen in training step
        user_item_history = get_item_history_from_user(user_id=user, data=train_data)
        
        # get predictions based on user history
        y_test_pred_items = model.single_predict(X=user_item_history, n=max(k_list))
            
        # get subset of ground truth for specific user
        y_test_true_items = get_item_history_from_user(user_id=user, data=test_data)
        
        # only consider users that have interactions in test set
        if y_test_true_items:
        
            # calculate hits for different k
            for k in k_list:
                
                # get sum of relevant items 
                hit_num = sum([item in y_test_true_items for item in y_test_pred_items[:k]])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(y_test_true_items))            
        
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]
    return precision, recall, f1

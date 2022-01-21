# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/hwwang55/DKN/blob/master/src/main.py

#import libraries 
import argparse
import numpy as np

# import custom code
from src.dkn_data_loader import load_data
from src.dkn import train, evaluate
from src.util.logger import setup_logging

np.random.seed(555)
logger = setup_logging(name=__file__, log_level='info')


parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=bool, default=True, help='whether to transform entity embeddings')
parser.add_argument('--use_context', type=bool, default=True, help='whether to use context embeddings')
parser.add_argument('--max_click_history', type=int, default=4, help='number of sampled click history for each user')
parser.add_argument('--n_filters', type=int, default=100, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of training epochs')

args = parser.parse_args()

logger.info('Loading data.')
train_data, val_data, test_data, random_test_data = load_data(args)

logger.info('Training model.')
train(args, train_data, val_data)

# logger.info('Evaluating model on complete test data.')
# evaluate(args, test_data)

# logger.info('Evaluating model on random test data.')
# evaluate(args, random_test_data)

# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/tezignlab/RippleNet-TF2/blob/master/main.py, which is under an MIT license.

# import libraries
import argparse
import numpy as np

# import custom code
from src.ripple_net import RippleNet


np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=48, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.03, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
parser.add_argument('--patience', type=int, default=8, help='early stop patience')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--test_set', choices=['complete', 'random', 'no_tfidf_ratings', 'no_word2vec_ratings', 'no_transformer_ratings'],
                    type=str, default='complete', 
                    help='the test set on which to evaluate the model')

args = parser.parse_args()
ripple_net = RippleNet(args)
ripple_net.train()
ripple_net.evaluate()

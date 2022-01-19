# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/hwwang55/DKN/blob/master/src/train.py

# import libraries
import os
import tensorflow as tf
import numpy as np

# import custom code
from src.dkn_model import DKN
from src.config import MODELS_DIR
from src.config import W2V_WORD_EMBEDDING_DIM, DKN_KGE_METHOD, DKN_KGE_ENTITY_EMBEDDING_DIM
from src.util.logger import setup_logging

tf.compat.v1.disable_eager_execution()
logger = setup_logging(name=__file__, log_level='info')


def get_feed_dict(model, data, start, end):
    feed_dict = {model.clicked_words: data.clicked_words[start:end],
                 model.clicked_entities: data.clicked_entities[start:end],
                 model.news_words: data.news_words[start:end],
                 model.news_entities: data.news_entities[start:end],
                 model.labels: data.labels[start:end]}
    return feed_dict


def train(args, train_data, val_data):
    model = DKN(args)
    saver = tf.compat.v1.train.Saver(max_to_keep=args.n_epochs)
    val_losses = []
    best_epoch = 0
    model_filepath = os.path.join(MODELS_DIR, 'dkn_model_' + str(W2V_WORD_EMBEDDING_DIM) + '_' + DKN_KGE_METHOD + '_' + str(DKN_KGE_ENTITY_EMBEDDING_DIM) + '.ckpt')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for step in range(args.n_epochs):
            # training
            start_list = list(range(0, train_data.size, args.batch_size))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size
                model.train(sess, get_feed_dict(model, train_data, start, end))

            # evaluation
            train_loss, train_auc, train_acc, train_micro_f1, train_macro_f1 = model.eval(sess, get_feed_dict(model, train_data, 0, train_data.size))
            val_loss, val_auc, val_acc, val_micro_f1, val_macro_f1  = model.eval(sess, get_feed_dict(model, val_data, 0, val_data.size))
   
            logger.info(
                    "Epoch: {:04d} \t train_loss= {:.5f} \t train_auc= {:.5f} \t train_acc= {:.5f} \t train_micro_f1= {:.5f} \t train_macro_f1= {:.5f} \t val_loss= {:.5f} \t val_auc= {:.5f} \t val_acc= {:.5f} \t val_micro_f1= {:.5f} \t val_macro_f1= {:.5f}".format(
                        step + 1, train_loss, train_auc, train_acc, train_micro_f1, train_macro_f1, val_loss, val_auc, val_acc, val_micro_f1, val_macro_f1)
                )

            val_losses.append(val_loss)

            # save the model if the validation loss decreased
            if val_losses[-1] == min(val_losses):
                best_epoch = step
                saver.save(sess, model_filepath)
        logger.info('Minimum validation loss of {} at epoch {}.'.format(min(val_losses), best_epoch))
            
def evaluate(args, test_data):
    model = DKN(args)
    model_filepath = os.path.join(MODELS_DIR, 'dkn_model_' + str(W2V_WORD_EMBEDDING_DIM) + '_' + DKN_KGE_METHOD + '_' + str(DKN_KGE_ENTITY_EMBEDDING_DIM) + '.ckpt')
    
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # Restore model
        logger.info('Restoring trained model.')
        try:
            saver.restore(sess, model_filepath)
            logger.info('Model restored.')
        except FileNotFoundError:
            logger.info('Model checkpoint does not exist. The model might not be trained yet or the checkpoint is invalid.')

        test_loss, test_auc, test_acc, test_micro_f1, test_macro_f1 = model.eval(sess, get_feed_dict(model, test_data, 0, test_data.size))
        logger.info("test_loss= {:.5f} \t test_auc= {:.5f} \t test_acc= {:.5f} \t test_micro_f1= {:.5f} \t test_macro_f1= {:.5f}".format(
                    test_loss, test_auc, test_acc, test_micro_f1, test_macro_f1)
                )

import argparse
import pickle

import numpy as np
import tensorflow as tf
import os

from data_loader import load_data
from train import train

np.random.seed(555)

# Limit GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    parser.add_argument('--eval_ratio', type=float, default=0.2, help='ratio of dataset used for evaluation')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of dataset used for test')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--comment', type=str, default='using exact hyperparam as original one')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    cached_preprocessed_data_filename = "../data/movie/preprocessed_data_info_{}".format(args.n_memory)

    # Preprocess data info
    if os.path.exists(cached_preprocessed_data_filename):
        print("loaded from cache : {}".format(cached_preprocessed_data_filename))
        data_info = pickle.load(open(cached_preprocessed_data_filename, 'rb'))
        
    else:
        data_info = load_data(args)

        print("saved to cache : {}".format(cached_preprocessed_data_filename))
        pickle.dump(data_info, open(cached_preprocessed_data_filename, 'wb'))

    train(args, data_info=data_info, show_loss=False, config=config)

import tensorflow as tf
import numpy as np
from model import RippleNet

from logger import Logger
from datetime import datetime
from tqdm import tqdm

timestamp = str(datetime.timestamp(datetime.now()))
SESSION_LOG_PATH = "../log/{}/".format(timestamp)


def train(args, data_info, show_loss, config):

    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    logger = Logger()
    logger.create_session_folder(SESSION_LOG_PATH)
    logger.set_default_filename(SESSION_LOG_PATH + "log.txt")
    logger.log(str(args))   # Log training and model hyper parameters

    model = RippleNet(args, n_entity, n_relation)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)

        for step in range(args.n_epoch):

            np.random.shuffle(train_data)

            # training
            for i in tqdm(range(0, train_data.shape[0], args.batch_size)):

                _, loss = model.train(sess, _get_feed_dict(args, model, train_data, ripple_set, i, i + args.batch_size))

                if show_loss:
                    print('%.1f%% %.4f' % (i / train_data.shape[0] * 100, loss))
                    logger.log('%.1f%% %.4f' % (i / train_data.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc = _evaluation(sess, args, model, train_data, ripple_set)
            eval_auc, eval_acc = _evaluation(sess, args, model, eval_data, ripple_set)
            test_auc, test_acc = _evaluation(sess, args, model, test_data, ripple_set)

            # Save the variables to disk.
            saver.save(sess, SESSION_LOG_PATH + "models/epoch_{}".format(step))

            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            logger.log(
                'epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def _get_feed_dict(args, model, data, ripple_set, start, end):

    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]

    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]

    return feed_dict


def _evaluation(sess, args, model, eval_data, ripple_set):

    auc_list = []
    acc_list = []

    for i in tqdm(range(0, eval_data.shape[0], args.batch_size)):
        auc, acc = model.eval(sess, _get_feed_dict(args, model, eval_data, ripple_set, i, i + args.batch_size))
        auc_list.append(auc)
        acc_list.append(acc)

    return float(np.mean(auc_list)), float(np.mean(acc_list))

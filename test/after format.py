# %%[markdown]:
#  # Test RippleNet Result

# In[ ]:

import argparse
import collections
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from IPython.display import display
from ipywidgets import FloatProgress, IntProgress
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# In[1]:


TEST_CODE = "1561537537.634447"
CHOSEN_EPOCH = 8

MODEL_PATH = "../log/{}/models/epoch_{}".format(TEST_CODE, CHOSEN_EPOCH)
LOG_PATH = "../log/{}/log.txt".format(TEST_CODE)


# In[2]:


np.random.seed(555)


# In[ ]:


# Limit GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[3]:


# Logger.py


class Logger:

    def set_default_filename(self, filename):
        self.default_filename = filename

    def create_session_folder(self, path):
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("\n ===> Successfully created the directory %s \n" % path)

    def log(self, text):
        with open(self.default_filename, 'a') as f:
            f.writelines(text)
            f.write("\n")

    def save_model(self, model, filename):
        pickle.dump(model, open(filename, 'wb'))


# In[4]:


# Model.py


class RippleNet(object):

    def __init__(self, args, n_entity, n_relation):

        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def _build_inputs(self):

        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):

            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))

            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))

            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def _build_embeddings(self):

        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []

        for i in range(self.n_hop):

            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):

        o_list = []
        for hop in range(self.n_hop):

            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)

        return o_list

    def update_item_embedding(self, item_embeddings, o):

        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)

        return item_embeddings

    def predict(self, item_embeddings, o_list):

        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):

        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):

            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))

        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):

            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)

        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):

        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)

        auc = roc_auc_score(y_true=labels, y_score=scores)

        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    # ============ Custom test purpose ============
    def custom_eval(self, sess, feed_dict):

        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc, labels, scores, predictions


# In[5]:


# Dataloader.py


def load_data(args):

    train_data, eval_data, test_data, user_history_dict = _load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = _get_ripple_set(args, kg, user_history_dict)

    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def _load_rating(args):
    print('reading rating file ...')

    rating_file = '../data/' + args.dataset + '/ratings_final'

    if os.path.exists(rating_file + '.npy'):
        print("loaded from cache : {}.npy".format(rating_file))
        rating_np = np.load(rating_file + '.npy')

    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)

        print("saved to cache : {}.npy".format(rating_file))
        np.save(rating_file + '.npy', rating_np)

    return _dataset_split(rating_np, eval_ratio=args.eval_ratio, test_ratio=args.test_ratio)


def _dataset_split(rating_np, eval_ratio=0.2, test_ratio=0.2):
    print('splitting dataset ...')

    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)

    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    kg_file = '../data/' + args.dataset + '/kg_final'

    if os.path.exists(kg_file + '.npy'):
        print("loaded from cache : {}.npy".format(kg_file))
        kg_numpy = np.load(kg_file + '.npy')

    else:
        kg_numpy = np.loadtxt(kg_file + '.txt', dtype=np.int32)

        print("saved to cache : {}.npy".format(kg_file))
        np.save(kg_file + '.npy', kg_numpy)

    n_entity = len(set(kg_numpy[:, 0]) | set(kg_numpy[:, 2]))
    n_relation = len(set(kg_numpy[:, 1]))

    kg = _construct_kg(kg_numpy)

    return n_entity, n_relation, kg


def _construct_kg(kg_numpy):
    print('constructing knowledge graph ...')

    kg = collections.defaultdict(list)
    for head, relation, tail in kg_numpy:
        kg[head].append((tail, relation))

    return kg


def _get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # Creating dictionary with format :
    # {user : [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]}

    ripple_set = collections.defaultdict(list)
    for user in tqdm(user_history_dict):
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])

            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)

                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]

                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set


# In[6]:


# Train.py


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


# # Args

# In[7]:


class Args:

    def __init__(self):
        self.dataset = 'movie'
        self.dim = 16
        self.eval_ratio = 0.2
        self.test_ratio = 0.2
        self.n_hop = 2
        self.kge_weight = 0.01
        self.l2_weight = 1e-07
        self.lr = 0.02
        self.batch_size = 1024
        self.n_epoch = 10
        self.n_memory = 32
        self.item_update_mode = 'plus_transform'
        self.using_all_hops = True
        self.comment = "running normally"


args = Args()

# %%[markdown]:
# ## Load the knowledge graph

# In[8]:


# Main.py

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

# %%[markdown]:
# # Testing the model
# %%[markdown]:
# Separate the preprocessed data

# In[11]:


train_data = data_info[0]
eval_data = data_info[1]
test_data = data_info[2]
n_entity = data_info[3]
n_relation = data_info[4]
ripple_set = data_info[5]

# %%[markdown]:
# # Evaluate

# In[12]:


model = RippleNet(args, n_entity, n_relation)


# In[13]:


# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=None)

sess = tf.Session(config=config)
saver = tf.train.import_meta_graph(MODEL_PATH + ".meta")
saver.restore(sess, MODEL_PATH)

# %%[markdown]:
# ## Custom precision at K eval

# In[14]:


def generate_truth_dict():

    truth_dict = {}
    for rating in tqdm(train_data):
        user_id, movie_id, score = rating

        if user_id not in truth_dict:
            truth_dict[user_id] = []

        if score == 1:
            truth_dict[user_id].append(movie_id)

    for rating in tqdm(test_data):
        user_id, movie_id, score = rating

        if user_id not in truth_dict:
            truth_dict[user_id] = []

        if score == 1:
            truth_dict[user_id].append(movie_id)

    for rating in tqdm(eval_data):
        user_id, movie_id, score = rating

        if user_id not in truth_dict:
            truth_dict[user_id] = []

        if score == 1:
            truth_dict[user_id].append(movie_id)

    return truth_dict


truth_dict = generate_truth_dict()

# %%[markdown]:
# ### ==============

# In[18]:

def predict(sess, args, model, users, items):

    test_data = _preprocess_test_data(users, items)

    scores = []
    for i in range(0, len(test_data), args.batch_size):

        feed_dict = _get_feed_dict(args, model, test_data, ripple_set, i, i + args.batch_size)
        _, _, _, batch_scores, _ = model.custom_eval(sess, feed_dict)
        scores = np.concatenate((scores, batch_scores))

    return scores

def _preprocess_test_data(users, items):
    "Preprocess test data so ripplenet can do feed forward with the right format"

    cust_test_data = []
    for user in users:
        for item in items:
            cust_test_data.append([user, item, 0]) # The last 0 is a dummy value to match input format

    return np.array(cust_test_data)


# In[19]:


def get_suggestion(user, k):

    items = [i for i in range(0, 15440)]
    prediction = predict(sess, args, model, [user], items)
    score_item_pairs = [(prediction[i], i) for i in items]
    top_k_recommendations = sorted(score_item_pairs, reverse=True)[:k]

    return top_k_recommendations


def get_top_truth(user, k):
    return truth_dict[user] if user in truth_dict else []

# In[20]:

def check_precision(prediction, truth, k=10):
    intersect = _get_intersect_pred_truth(pred, truth, k)
    len_intersect = len(intersect)
    len_truth = len(truth) if 0 < len(truth) <= k else k

    return intersect, len_intersect / len_truth

def _get_intersect_pred_truth(pred, truth, k):
    pred_item_set = {x[1] for x in pred}
    truth_item_set = set(truth)

    return pred_item_set.intersection(truth_item_set)


# In[21]:

k_suggestion = 10
n_users = 10

sample_user = np.random.randint(1, 15000, n_users) # sampling
# sample_user = [i in range(0, 15000)] # uncomment to use non sampling

suggested_items = []
truth_items = []
intersects = []
scores = []

all_intersect = None
all_union = None

for user in tqdm(sample_user):

    try:

        top_suggested_items = get_suggestion(sample_user, k_suggestion)
        top_truth_items = get_top_truth(sample_user, k_suggestion)

        intersect, score = check_precision(top_suggested_items, top_truth_items, k=k_suggestion)

        suggested_items.append(top_suggested_items)
        truth_items.append(top_truth_items)
        intersects.append(intersect)
        scores.append(score)

        if all_intersect is None:
            all_intersect = top_suggested_items
        else:
            all_intersect = all_intersect.intersection(top_suggested_items)

        if all_union is None:
            all_union = top_suggested_items
        else:
            all_union = all_union.union(top_suggested_items)

    except Exception as e:
        print("error occur for {} : {}".format(user, e))
        

# In[23]:
print("Prec@k score:", np.average(scores))
# print("top_suggested_items:", top_suggested_items)
# print("truth_items:", truth_items)

print("\nintersect")
print(all_intersect, len(all_intersect))
print("\nunion")
print(all_union, len(all_union))
print("\ndistinct rate")
print((len(all_union)) / (n_users * k_suggestion))

# In[23]:

sample_user = [np.random.randint(1, 138000) for i in range(0, 3)]

for user in sample_user:

    prediction = get_suggestion(user, 10)
    truth = get_top_truth(user, 10)

    display(user)
    display((prediction))
    display([x[1] for x in prediction])
    display((truth))
    display(check_precision(prediction, truth, 10))
    display("==================")
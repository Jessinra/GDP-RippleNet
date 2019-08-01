import collections
import os
import numpy as np
from tqdm import tqdm


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

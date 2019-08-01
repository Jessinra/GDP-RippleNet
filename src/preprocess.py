import argparse
import numpy as np
from tqdm import tqdm

RATING_FILENAME = "ratings_re2.csv"
ITEMS_FILENAME = "moviesIdx2.txt"
TRIPLES_FILENAME = 'triples_idx2.txt'
THRESHOLD = 4

OUTPUT_RATING_FILENAME = 'ratings_final.txt'
OUTPUT_KG_FILENAME = 'kg_final.txt'

np.random.seed(5325)


def convert_rating():

    print('reading item file ...')
    item_set = _read_item_set(ITEMS_FILENAME)

    print('reading rating file ...')
    users_positive_items, users_negative_items = _read_ratings_groupby_user(RATING_FILENAME)

    print('converting rating file ...')
    _write_formatted_ratings(OUTPUT_RATING_FILENAME, item_set, users_positive_items, users_negative_items)

    print('converting rating file success !')


def _read_item_set(items_filename):
    items = open(items_filename, encoding='utf-8').readlines()
    return set(range(len(items)))


def _read_ratings_groupby_user(rating_filename):
    user_ratings = open(rating_filename, encoding='utf-8').readlines()
    users_positive_items, users_negative_items = _split_ratings_groupby_user(user_ratings)
    return users_positive_items, users_negative_items


def _split_ratings_groupby_user(user_ratings):

    users_positive_items = dict()
    users_negative_items = dict()

    for line in tqdm(user_ratings):

        array = line.strip().split(',')
        user_index = int(array[0])
        item_index = int(array[1])
        rating = float(array[2])

        dict_to_put = users_positive_items if rating >= THRESHOLD else users_negative_items
        if user_index not in dict_to_put:
            dict_to_put[user_index] = set()
        dict_to_put[user_index].add(item_index)

    return users_positive_items, users_negative_items


def _write_formatted_ratings(output_rating_filename, item_set, users_positive_items, users_negative_items):

    writer = open(output_rating_filename, 'w', encoding='utf-8')
    for user_index, pos_item_set in tqdm(users_positive_items.items()):

        # Write positive sample
        for item in (pos_item_set):
            writer.write("{}\t{}\t1\n".format(user_index, item))

        # ! Negative sample using unwatched instead of negative rated movies !
        unwatched_set = item_set - pos_item_set
        if user_index in users_negative_items:
            unwatched_set -= users_negative_items[user_index]

        # Write negative sample (unwatched)
        for item in (np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False)):
            writer.write("{}\t{}\t0\n".format(user_index, item))

    writer.close()


def convert_kg():

    with open(OUTPUT_KG_FILENAME, 'w', encoding='utf-8') as writer:

        raw_knowledge_graph = open(TRIPLES_FILENAME, encoding='utf-8')
        for line in raw_knowledge_graph:
            head, relation, tail = line.strip().split(' ')
            writer.write("{}\t{}\t{}\n".format(head, relation, tail))

    print('converting kg file success !')


if __name__ == "__main__":

    convert_rating()
    convert_kg()

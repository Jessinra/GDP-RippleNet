import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings_re.csv', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt'})
ITEMS_FILE_NAME = dict({'movie': 'movies_re.csv'})
SEP = dict({'movie': ',', 'book': ';', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})


def convert_rating():

    print('reading rating file ...')
    items_filename = '../data/' + DATASET + '/' + ITEMS_FILE_NAME[DATASET]
    items = open(items_filename, encoding='utf-8').readlines()
    item_set = set(range(len(items)))

    user_pos_ratings = dict()
    user_neg_ratings = dict()

    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    for line in open(file, encoding='utf-8').readlines()[1:]:

        array = line.strip().split(SEP[DATASET])
        user_index = int(array[0])
        item_index = array[1]
        rating = float(array[2])

        # Separate positive & negative rated items
        if rating >= THRESHOLD[DATASET]:
            if user_index not in user_pos_ratings:
                user_pos_ratings[user_index] = set()
            user_pos_ratings[user_index].add(item_index)

        else:
            if user_index not in user_neg_ratings:
                user_neg_ratings[user_index] = set()
            user_neg_ratings[user_index].add(item_index)

    print('reading rating file success !')
    print('converting rating file ...')

    # Output file
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    for user_index, pos_item_set in user_pos_ratings.items():

        # Write positive sample
        for item in pos_item_set:
            writer.write("{}\t{}\t1\n".format(user_index, item))

        # ! Negative sample using unwatched instead of negative rated movies !
        unwatched_set = item_set - pos_item_set
        if user_index in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index]

        # Write negative sample (unwatched)
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write("{}\t{}\t0\n".format(user_index, item))

    writer.close()
    print('converting rating file success !')


def convert_kg():
    print('converting kg file ...')

    # Output file
    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    if DATASET == 'movie':
        raw_knowledge_graph = open('../data/' + DATASET + '/triples_idx.txt', encoding='utf-8')
    else:
        raw_knowledge_graph = open('../data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8')

    for line in raw_knowledge_graph:
        head, relation, tail = line.strip().split(' ')
        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('converting kg file success !')


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    convert_rating()
    convert_kg()

    print('done')

# -*- coding: utf-8 -*-
import logging
from sklearn.manifold import TSNE as tsne
import matplotlib.pyplot as plt
import Word2Vec
from gensim.models import word2vec
import pandas as pd
import parser


def print_word(model):
    plt.rc('font', family='verdana')
    tsne_model = tsne()
    vec = tsne_model.fit_transform(model.syn0)
    for i, txt in enumerate(model.index2word):
        plt.annotate(txt, vec[i])

    plt.show()


if __name__ == '__main__':
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    train_data = Word2Vec.read_train_data('../data')

    # Test data first 50 texts from each folder

    test_data = Word2Vec.read_test_data('../data')

    model_name = '300size_20min_count.model'
    Word2Vec.train_and_save_model(model_name, train_data, min_count=30)
    model = word2vec.Word2Vec.load(model_name)
    forest = Word2Vec.trainRandomForest(train_data, test_data, model)

    print_word(model)

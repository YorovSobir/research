# -*- coding: utf-8 -*-
import os
from parser import pdf_to_text, text_to_sentences, sentence_to_wordlist
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from gensim.models import word2vec


def makeFeatureVec(words, model, size=300):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((size,), dtype="float")

    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords += 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(texts, model, size=300):
    # Given a set of texts (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    textFeatureVecs = np.zeros((len(texts), size), dtype="float")
    #
    # Loop through the reviews
    for text in texts:
        # Call the function (defined above) that makes average feature vectors
        textFeatureVecs[counter] = makeFeatureVec(text, model, size)
        #
        # Increment the counter
        counter += 1.
    return textFeatureVecs


def train_and_save_model(model_name, train_data, workers=4,
                         size=300, min_count=20,
                         window=10, sample=1e-3):
    sentences = []
    for text in train_data['text']:
        sentences += text_to_sentences(text)

    model = word2vec.Word2Vec(sentences, workers=workers,
                              size=size, min_count=min_count,
                              window=window, sample=sample)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    return model_name


def read_train_data(data_path):
    texts = []
    positive = data_path + '/positive/'
    print 'get data from folder ' + positive

    for filename in os.listdir(positive):
        text = pdf_to_text(os.path.join(positive, filename), True)
        texts.append([filename, text, 1])

    negative = data_path + '/negative/'
    print 'get data from folder ' + negative

    for filename in os.listdir(negative):
        text = pdf_to_text(os.path.join(negative, filename), True)
        texts.append([filename, text, 0])

    header = ['id', 'text', 'classification']
    train = pd.DataFrame(texts, columns=header)
    return train


def read_test_data(data_path):
    texts = []
    positive = data_path + '/positive/'
    print 'get test data from folder ' + positive
    i = 1
    for filename in os.listdir(positive):
        text = pdf_to_text(os.path.join(positive, filename), True)
        texts.append([filename, text, 1])
        i += 1
        if i == 51:
            break

    negative = data_path + '/negative/'
    print 'get test data from folder ' + negative
    i = 1
    for filename in os.listdir(negative):
        text = pdf_to_text(os.path.join(negative, filename), True)
        texts.append([filename, text, 0])
        i += 1
        if i == 51:
            break

    header = ['id', 'text', 'classification']
    test = pd.DataFrame(texts, columns=header)
    return test


def trainRandomForest(train_data, test_data, model, size=300):

    clean_train = []
    for text in train_data['text']:
        clean_train.append(sentence_to_wordlist(unicode(text, 'utf-8')))
    trainDataVecs = getAvgFeatureVecs(clean_train, model, size)

    clean_test = []
    for text in test_data['text']:
        clean_test.append(sentence_to_wordlist(unicode(text, 'utf-8')))
    testDataVecs = getAvgFeatureVecs(clean_test, model, size)


    # Fit a random forest to the training data, using 100 trees

    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train_data['classification'])

    importances = forest.feature_importances_

    #
    # Test & extract results
    # result = forest.predict(testDataVecs)


        # Plot the feature importances of the forest
    # Write the test results
    print forest.score(testDataVecs, test_data['classification'])

    return forest


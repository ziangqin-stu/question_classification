import nltk
import random
import os
import pickle
import requests
import json


def corpus_analysis(filename):
    f = open('./data/qc/' + filename, 'r')

    corpus = []
    line = f.readline().lower()
    while len(line) > 0:
        lable = line.split(' ')[0]
        sentence_list = line.split(' ')[1:]
        sentence_list[-1] = sentence_list[-1].split('\n')[0]
        corpus.append((sentence_list, lable))
        line = f.readline().lower()
    return corpus


def read_dictionary():
    dictionary = []
    f = open('./data/qc/words_alpha.txt', 'r')
    line = f.readline()
    line = line.split('\n')[0]
    while len(line)>0:
        dictionary.append(line)
        line = f.readline()
        line = line.split('\n')[0]
    f.close()
    return dictionary


def onehot_encoder(sentence_list, dictionary):
    result = []
    for elem in sentence_list:
        if elem in dictionary:
            result.append(dictionary.index(elem))
    return ' '.join(str(elem) for elem in result)


def sentence_features(sentence_list, dictionary):
    features = {}
    features['first_word'] = sentence_list[0]
    features['last_word'] = sentence_list[-1]
    features['length'] = len(sentence_list)
    features['onehot'] = onehot_encoder(sentence_list, dictionary)
    return features


def fol_features(fol, dictionary):
    features = {}
    features['has_what'] = 'what' in fol
    features['has_where'] = 'where' in fol
    features['has_witch'] = 'witch' in fol
    features['has_when'] = 'when' in fol
    features['has_if'] = 'if' in fol
    features['has_can'] = 'can' in fol
    features['length'] = len(fol)
    fol_list = nltk.word_tokenize(fol)
    features['onehot'] = onehot_encoder(fol_list, dictionary)
    return features

def nl2fol(sentence_list):
    sentence = ' '.join(sentence_list)
    url = 'http://192.168.38.94:8115/semantic-parse'
    data = {'query': sentence}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    if 'FOL' in response.json():
        return response.json()["FOL"]
    else:
        return "(" + sentence + ")"


def read_features(rebuild, file='train-quarter.txt', featureGenerator = 'normal'):
    if rebuild == False and os.path.exists('fol_features.pickle'):
        featuresets = pickle.load(open('fol_features.pickle', 'rb'))
    else:
        # initialize data
        corpus = corpus_analysis(file)
        dictionary = read_dictionary()
        # nls2fol
        fol_corpus = []
        print('    1.1')
        for elem in corpus:
            fol_corpus.append((nl2fol(elem[0]), elem[1]))
        # create feature_set
        print('    1.2')
        featuresets = [(fol_features(fol, dictionary), label) for (fol, label) in fol_corpus]
        # store feature_set
        pickle.dump(featuresets, open('fol_features.pickle', 'wb'))
    return featuresets


def read_model(rebuild, iter):
    if rebuild == False and os.path.exists('maxent-fol_classifier.pickle'):
        f = open('maxent-fol_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    else:
        # train classifier
        classifier = nltk.MaxentClassifier.train(train_set, max_iter=iter)
        # save model
        f = open('maxent-fol_classifier.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()
    return classifier


def run(maxIter, trainFile):
    # create feature_set
    print('1')

    featuresets = read_features(rebuild=True, file=trainFile)
    print('2')

    random.shuffle(featuresets)
    trainSize = int(len(featuresets) / 10)
    train_set, test_set = featuresets[trainSize:], featuresets[:trainSize]
    print('3')

    classifier = read_model(rebuild=False, iter=maxIter)
    print('4')

    # test cases
    print(nltk.classify.accuracy(classifier, test_set))

    dictionary = read_dictionary()
    print(classifier.classify(fol_features("(what(is<name,?>))", dictionary)))


# _______________Main_______________

# run(maxIter=10, trainFile='train-quarter.txt')
# run(maxIter=6, trainFile='train.txt')
run(maxIter=15+, trainFile='train.txt')

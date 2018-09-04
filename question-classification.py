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
    dict = []
    f = open('./data/qc/words_alpha.txt', 'r')
    line = f.readline()
    line = line.split('\n')[0]
    while len(line)>0:
        dict.append(line)
        line = f.readline()
        line = line.split('\n')[0]
    f.close()
    return dict


def onehot_encoder(sentence_list, dict):
    result = []
    for elem in sentence_list:
        if elem in dict:
            result.append(dict.index(elem))
    return ' '.join(str(elem) for elem in result)


def sentence_features(sentence_list, dict):
    features = {}
    features['first_word'] = sentence_list[0]
    features['last_word'] = sentence_list[-1]
    features['length'] = len(sentence_list)
    features['onehot'] = onehot_encoder(sentence_list, dict)
    return features


# _______________Main_______________

url = 'http://192.168.38.94:8115/semantic-parse'
data = {'query':'What is myopia ?'}
headers = {'Content-Type': 'application/json'}
response = requests.post(url=url, headers=headers, data=json.dumps(data))
FOL = response.json()["FOL"]

# initialize data
corpus = corpus_analysis('train.txt')
dict = read_dictionary()

# create feature_set
featuresets = [(sentence_features(sentence_list, dict), label) for (sentence_list, label) in corpus]
random.shuffle(featuresets)
train_set, test_set = featuresets[500:], featuresets[:500]

if not os.path.exists('maxent_classifier.pickle'):
    # train classifier
    classifier = nltk.MaxentClassifier.train(train_set, max_iter=5)

    # save model
    f = open('maxent_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
else:
    f = open('maxent_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()


# test cases
print(nltk.classify.accuracy(classifier, test_set))

r = requests.post('http://192.168.38.94:8115/semantic-parse', data = {"query":"What is myopia ?"})



print(classifier.classify(sentence_features("How long is the car ?".split(' '), dict)))
print(classifier.classify(sentence_features("Who likes apple?".split(' '), dict)))
print(classifier.classify(sentence_features("What is this?".split(' '), dict)))
print(classifier.classify(sentence_features("Does the man likes apple?".split(' '), dict)))
print(classifier.classify(sentence_features("Is the car yours?".split(' '), dict)))

# _______________Error-Set_______________

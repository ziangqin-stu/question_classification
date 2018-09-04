import nltk
from nltk.corpus import names
import random


def gender_features(word):
    features = {'last_letter':word[-1], 'first_letter':word[0], 'name':str(word), 'len': len(word)}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = word.lower().count(letter)
        features["has({})".format(letter)] = (letter in word.lower())
    return features

# _______________main_______________

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.MaxentClassifier.train(train_set)
print(classifier.classify(gender_features('Neo')))
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
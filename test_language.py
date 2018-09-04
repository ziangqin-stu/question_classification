import nltk
import pickle
import os

b = ['hello']
if os.path.exists('variable_a.txt'):
    b = pickle.load(open('variable_a.txt', 'rb'))
    a = b
else:
    a = ('a', 123)
    pickle.dump(a, open('variable_a.txt', 'wb'))

print(a, b)

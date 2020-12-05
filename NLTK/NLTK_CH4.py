import numpy as np
import pandas as pd
import matplotlib
import nltk,re, pprint
import os

words = ['I', 'turned', 'off', 'the', 'spectroroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
list(zip(words, tags))
list(enumerate(words))

#Spliting Train-Test
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
text == training_data + test_data

#funcion
def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'

tag(str("noun"))
tag('the')

def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.
    Given a list of reference values and a corresponding list of test values,\n
    return the fraction of corresponding values that are equal.\n
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.
    >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
    0.5
    @param reference: An ordered list of reference values.\n
    @type reference: C{list}\n
    @param test: A list of values to compare against the corresponding\n
    reference values.\n
    @type test: C{list}\n
    @rtype: C{float}\n
    @raise ValueError: If C{reference} and C{length} do not have the
    same length.
    """
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in izip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)

accuracy



#getting loaction of NLTK files
nltk.metrics.__file__


from timeit import Timer
vocab_size = 100000
setup_list = "import random; vocab = range(%d)" % vocab_size
setup_set = "import random; vocab = set(range(%d))" % vocab_size
statement = "random.randint(0, %d) in vocab" % vocab_size * 2
print(Timer(statement, setup_list).timeit(1000))
print(Timer(statement, setup_set).timeit(1000))














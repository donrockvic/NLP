#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:37:37 2019

@author: vicky
"""

import numpy as np
import pandas as pd
import matplotlib
import nltk,re, pprint

#Tokenization and tagging
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

#Documenations
nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('NN.*')

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

#Similar method takes a word w, finds all contexts w 1 w w 2 ,then finds all words w' that appear in the same context
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')
text.similar('bought')
text.similar('over')
text.similar('the')

#Representing Tagged tokens
#NLTK Functions
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
sent = '''
    The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
     other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
     Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
     said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
     accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
     interest/NN of/IN both/ABX governments/NNS ''/'' ./.
     '''
[nltk.tag.str2tuple(t) for t in sent.split()]
nltk.corpus.brown.tagged_words()
#nltk.corpus.brown.tagged_words(simplify_tags=True)
test = nltk.corpus.indian.tagged_words()
test[140:150]
nltk.corpus.brown.tagged_sents()

from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
len(tag_fd.keys())
tag_fd.plot(10)


#Automatic Tagging
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()

#Assigning a Defalt tag NN to everything
#Default Tagger
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)

#Regular Expression Tagger
patterns = [
        (r'.*ing$', 'VBG'),     #gerunds
        (r'.*ed$', 'VBD'),      #simple past
        (r'.*es$', 'VBZ'),      #3rd singular present
        (r'.*ould$', 'MD'),     #modals
        (r'.*\'s$', 'NN$'),     #possessive nouns
        (r'.*s$', 'NNS'),       #plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),    #cardinal numbers
        (r'.*', 'NN')           #nouns (default)
        ]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

#When noe of the tagger is assigned we will go with default
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = list(fd.keys())[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

#Backoff
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                     backoff=nltk.DefaultTagger('NN'))



def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

display()


#UNIGRAM Tagging
#training and testing
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)  #training
unigram_tagger.tag(nltk.word_tokenize("My Name is Vicky. I love a girl name Diksha"))


#Training and testing by splitting the dataset
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

#Bigram Tagger
bigram_tagger = nltk.BigramTagger(train_sents)  #Training
bigram_tagger.tag(nltk.word_tokenize("My Name is Vicky. I love a girl name Diksha"))


#Complicated Algo i.e., multiple tagger
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents,cutoff=2, backoff=t1)
t2.tag(nltk.word_tokenize("My Name is Vicky. I love a girl name Diksha"))

#saving the tagged data
from cloudpickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

#opening the saved one
from cloudpickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

#testing it
text = """The board's action shows what free enterprise
is up against in our complex maze of regulatory laws ."""
tokens = text.split()
tagger.tag(tokens)
































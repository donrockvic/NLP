import numpy as np
import pandas as pd
import matplotlib
import nltk
#loads some text to explorer
#nltk.corpus.gutenberg.fileids()
#emma = nltk.corpus.gutenberg.words('austen-emma.txt')

from nltk.corpus import gutenberg
gutenberg.fileids()
emma = gutenberg.words('austen-emma.txt')
len(emma)
gutenberg.abspath(emma)

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab),fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences

#Webtext corpus
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid)

#Chat corpus
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

#brown corpous
from nltk.corpus import brown
brown.categories()

from nltk.corpus import brown
news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m])

cfd = nltk.ConditionalFreqDist(
(genre, word)
for genre in brown.categories()
for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

#reuters corpus
from nltk.corpus import reuters
reuters.fileids()
reuters.categories()
reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids('barley')
reuters.fileids(['barley', 'corn'])

#inaugural corpus
from nltk.corpus import inaugural
inaugural.fileids()
cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target))
cfd.plot()

#other language corpus
nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.fileids()
nltk.corpus.udhr.words('Javanese-Latin1')[11:]

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
(lang, len(word))
for lang in languages
for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
cfd.conditions()

#Loading your own corpus
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('connectives')

cfd.tabulate(conditions=['English', 'German_Deutsch'],samples=range(10), cumulative=True)

#pairing word from a list
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven','and', 'the', 'earth', '.']
data = list(nltk.bigrams(sent))
data

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word),
        word = cfdist[word].max()


text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
print(cfd['living'])
generate_model(cfd, 'living')


#sigular to plural
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

plural('axe')

#Wordlist Corpora
#Finding a english Test
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
unusual_words(nltk.corpus.nps_chat.words())

#Stopword Corpus
from nltk.corpus import stopwords
stopwords.words('english')

#function that compute what fraction of word are not stopword
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())

#solving a word puzzle
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters]


#Name Corpus
names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

cfd = nltk.ConditionalFreqDist(
        (fileid, name[-1])
        for fileid in names.fileids()
        for name in names.words(fileid))
cfd.plot()


#A Pronouncing Dictionary
#CMU Pronouncing Dictionary for U.S. English
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[39943:39951]:
    print(entry)

syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]


def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]
[w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]

prondict = nltk.corpus.cmudict.dict()
prondict['fire']
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
prondict['blog']

#Different language comapritive corupys
from nltk.corpus import swadesh
swadesh.fileids()
#french and english
fr2en = swadesh.entries(['fr', 'en'])
fr2en


#wordnet
#synonyms
from nltk.corpus import wordnet as wn
wn.synsets('pure')              #synset -- synmnym set
wn.synset('pure.s.04').lemmas()
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()
wn.synset('car.n.01').lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()
wn.synsets('car')
wn.lemmas('car')

nltk.app.wordnet()

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[26]
sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()])
motorcar.hypernyms()
paths = motorcar.hypernym_paths()
len(paths)
[synset.name() for synset in paths[0]]
motorcar.root_hypernyms()


#parts of Heriarchy
wn.synset('tree.n.01').part_meronyms()
wn.synset('tree.n.01').substance_meronyms()
wn.synset('tree.n.01').member_holonyms()

for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())











































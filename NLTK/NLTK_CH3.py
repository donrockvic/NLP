import numpy as np
import pandas as pd
import matplotlib
import nltk,re, pprint
import os
os.listdir('.')
#Processing RAW TEXT

#opening a online corpous
from urllib.request import urlopen

#Reading Text
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
raw = urlopen(url).read()
type(raw)
len(raw)
raw[:75]
tokens = nltk.word_tokenize(raw.decode("utf-8"))
type(tokens)
len(tokens)
tokens[:10]
text = nltk.Text(tokens)
type(text)
text[1020:1060]
text.collocations()   #not working
raw = raw[5303:1157681]
raw.find(b'PART I')
raw.rfind(b"End of Project Gutenberg's Crime")


#Reading HTML
from bs4 import BeautifulSoup
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
html[:60]
print(html)
raw = BeautifulSoup(html)
type(raw.prettify())
tokens = nltk.word_tokenize(raw.prettify())
tokens
tokens = tokens[96:399]
text = nltk.Text(tokens)
text.concordance('gene')

#RSSFeed blog parser
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
len(llog.entries)
post = llog.entries[2]
post.title
content = post.content[0].value
content[:70]
nltk.word_tokenize(BeautifulSoup(content).prettify())
nltk.word_tokenize(BeautifulSoup(llog.entries[2].content[0].value).prettify())

#Reading local file
#using python
f = open('document.txt', 'r')
f.read()
for line in f:
    print(line.strip())

#using NLTK
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'r').read()

path = nltk.data.find(resource_name=u'document.txt')
raw = open(path, 'r').read()
raw

#Capturing User Input
s = input("Enter some text: ")
print("You typed", len(nltk.word_tokenize(s)), "words.")
s

#String Manupulation
#when string goes over multiple lines
couplet = "Shall I compare thee to a Summer's day?"\
"Thou are more lovely and more temperate:"
#OR
couplet = ("Rough winds do shake the darling buds of May,"
"And Summer's lease hath all too short a date:")
print(couplet)


#when we have to get a new line
couplet = """Shall I compare thee to a Summer's day?
 Thou are more lovely and more temperate:"""
print(couplet)

#exercise
a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b:
    print(b)


#Counting INDIViDUAL character
from nltk.corpus import gutenberg
raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
fdist.keys()
fdist.plot()
help(str)

#Extracting Encoded Text from Files
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
import codecs
f = codecs.open(path, encoding='latin2')
print(f.read())

#finding Integer Ordinal
ord('$')

#The module unicodedata lets us inspect the properties of Unicode characters
import unicodedata
lines = codecs.open(path, encoding='latin2').readlines()
line = lines[2]
print(line.encode('unicode_escape'))
for c in line:
    if ord(c) > 127:
        print('%r U+%04x %s' % (c.encode('utf-8'), ord(c), unicodedata.name(c)))


#Matching with regular expression
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
wordlist
[w for w in wordlist if re.search('ed$', w)]
[w for w in wordlist if re.search('^..j..t..$', w)]
sum(1 for w in text if re.search('^e-?mail$', w))
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]



#Searching Tokenized Text
from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a> (<.*>) <man>")

chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*> <.*> <bro>")

from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")

#Normalizing Text using NLTK
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
#Stemmers : Removing Affixes
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]


#Indexing a text using a stemmer.
class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))


    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text
text.concordance('lie')

#Lemmatization
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]


#Regular Expressions for Tokenizing Text
raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
 though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
 well without--Maybe it's always pepper that makes people hot-tempered,'..."""
re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)
re.split(r'\s+', raw)
re.split(r'\W+', raw)
'xx'.split('x')
#final
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))

#Tokenization using NLTK
text = 'That U.S.A. poster-print costs $12.40...'
# set flag to allow verbose regexp
# abbreviations, e.g. U.S.A.
# words with optional internal hyphens
# currency and percentages, e.g. $12.40, 82%
# these are separate tokens
# ellipsis
pattern = r'''(?x)# set flag to allow verbose regexps
...([A-Z]\.)+# abbreviations, e.g. U.S.A.
...| \w+(-\w+)*# words with optional internal hyphens
...| \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
...| \.\.\.# ellipsis
...| [][.,;"'?():-_`] # these are separate tokens
... '''
nltk.regexp_tokenize(text,r"\[A-Z]\.+(?:[-']\w+)*|'|[-.(]+|\S\w*")

#Sentence Segmentation
sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)
pprint.pprint(sents[171:181])


#Word Segmentation
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size

from random import randint
def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0,len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, int(round(temperature)))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
anneal(text, seg1, 5000, 1.2)


#Wrapping a Text output
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',','more', 'is', 'said', 'than', 'done', '.']
for word in saying:
    print(word, '(' + str(len(word)) + '),',)


from textwrap import fill
format = '%s (%d),'
pieces = [format % (word, len(word)) for word in saying]
output = ' '.join(pieces)
wrapped = fill(output)
print(wrapped)


























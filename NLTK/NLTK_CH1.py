import numpy as np
import pandas as pd
import matplotlib
import nltk
#loads some text to explorer
from nltk.book import *
print(text1)

text1.concordance("monstrous")
text3.concordance("lived")

#find the othe words which appears in the same context
text1.similar("monstrous")

#list most frequest common contexts first
text2.common_contexts(["monstrous", "very"])

#display a plot showing the distribution
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

#print or generate trandom text
text3.generate()

len(text3)
sorted(set(text3))
len(set(text3))
text1.count('heaven')

#gwtting word using index
text4[173]
text4.index('awaken')


#lexical richness: How many time a word is used
len(text3) / len(set(text3))
text3.count("smote")
#function for lexical richness
def lexical_diversity(text):
    return len(text) / len(set(text))
#function for counting the % for a word
def percentage(count, total):
    return 100 * count / total

lexical_diversity(text3)
percentage(text4.count('the'), len(text4))

#Frequency Distribution
fdist1 = FreqDist(text1)
fdist1
fdist1['whale']
fdist1.plot(50, cumulative=True)
fdist1.tabulate()
#Returns all words that occurs only once
fdist1.hapaxes()
#getting keys
vocabulary1 = list(fdist1.keys())
vocabulary1[:50]

#Finding all the word above length 15
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])

#Bigrams
deel=list(bigrams(['more', 'is', 'said', 'than', 'done']))
deel

#collaction
print(', '.join(text4.collocation_list()))

#frequency of length of word
fdist = FreqDist([len(w) for w in text1])
fdist
fdist.keys()
fdist.items()
fdist.max()
fdist.freq(3)











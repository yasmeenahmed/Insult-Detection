# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:48:12 2018

@author: Yasmeen
"""
import random
import functools
from nltk.util import skipgrams
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#text = [word_tokenize(line.strip()) for line in open("metamorphosis_clean.txt" , "r")]
text = [['This', 'is', 'a', 'text', 'block', 'start', '.'], ['This', 'is', 'the', 'end', '.']]
#print(text)
text  = ["this is a text" , "this is a text","this is a text", "this is a text","this is a text"]
# Example of a bigram with k=2 skips.
skipper = functools.partial(skipgrams, n=3 , k=2)
#print(list(skipper(text[0])))
#initialize the vectorizer with skipgrams analayzer
vectorizer = CountVectorizer(ngram_range=(1,2),stop_words="english",lowercase=True) 
vectorizer.fit(text)
print(vectorizer.vocabulary_)
"""
vectorizer2 = CountVectorizer(ngram_range=(1,5),stop_words="english",lowercase=True,analyzer=skipper) 
vectorizer2.fit(text)
if vectorizer.vocabulary_ == vectorizer2.vocabulary_:
    print("t")
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:31:55 2018

@author: Yasmeen
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing off stopwords filtration."
stop_words = set(stopwords.words("English"))
words = word_tokenize(example_sentence)
filtered_sentence  = []
"""
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
"""
filtered_sentence = [w for w in words if not w in stop_words]
        
print(filtered_sentence)

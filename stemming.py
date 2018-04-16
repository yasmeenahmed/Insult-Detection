# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:45:22 2018

@author: Yasmeen
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
"""
example_words = ["python","pythoner","pythoning","running0","ran","pythonoly","sent"]
for w in example_words:
    print(ps.stem(w))
"""

new_text = "it's very important to pythonly while you are pythoning with pyhton. All pythoners have pythoned poorly at least once"
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))


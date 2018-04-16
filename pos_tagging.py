# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:47:45 2018

@author: Yasmeen
not compelete
"""
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#unserpervised algo
train_text = state_union.raw("metamorphosis_clean.txt")
sample_text = state_union.raw("metamorphosis_clean.txt")
custom_sent _tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i  in tokenized:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
        
    except Exception as e:
        print(str(e))
        
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:51:37 2018

@author: Yasmeen
"""

from nltk.tokenize import word_tokenize, sent_tokenize
example_text = "Hello Mr. Smith, how are you doing? the wether is great and pyhton is awesome"
example = ['Mary had a little lamb' , 
            'Jack went up the hill' , 
         'Jill followed suit' ,    
            'i woke up suddenly' ,
            'it was a really bad dream...']
tokenized_sents = [word_tokenize(i) for i in example]
for i in tokenized_sents:
    for j in i:
        print(j)
        
print(" ".join(tokenized_sents[1]))
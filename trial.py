from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, ensemble, linear_model, svm
from numpy import log, ones, array, zeros, mean, std, repeat
import numpy as np
import scipy.sparse as sp
import re
import csv
from time import time

DIR_PATH = "C://workspace//impermium//"

TRAIN_FILE      = DIR_PATH + "train.csv"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"   # This is also used for training, together with TRAIN_FILE
BADWORDS_FILE   = DIR_PATH + "badwords.txt"              # attached with submission  

TEST_FILE       = DIR_PATH + "verification.csv"          # set this to the new test file name
PREDICTION_FILE = DIR_PATH + "preds.csv"                 # predictions will be written here 

def normalize(f):
    f = [x.lower() for x in f]
    f = [x.replace("\\n"," ") for x in f]        
    f = [x.replace("\\t"," ") for x in f]        
    f = [x.replace("\\xa0"," ") for x in f]
    f = [x.replace("\\xc2"," ") for x in f]

    f = [x.replace(" u "," you ") for x in f]
    f = [x.replace(" em "," them ") for x in f]
    f = [x.replace(" da "," the ") for x in f]
    f = [x.replace(" yo "," you ") for x in f]
    f = [x.replace(" ur "," you ") for x in f]
    
    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]

    bwMap = loadBW()
    for key, value in bwMap.items():
        kpad = " " + key + " "
        vpad = " " + value + " "
        f = [x.replace(kpad, vpad) for x in f]
    #some code for stemming
    return f
#some code for n grams (use tdifvectorizer)
def ngrams(data , labels, ntrain, mn=1, mx=1, binary = False, stopwords, donorm = False, verbose = True, analyzer_char = False):
    f = data
    if donorm:
        f = normalize(f)
    
    ftrain = f[:ntrain]
    ftest  = f[ntrain:]
    y_train = labels[:ntrain]
    t0 = time()
       analyzer_type = 'word'
    if analyzer_char:
        analyzer_type = 'char'
        





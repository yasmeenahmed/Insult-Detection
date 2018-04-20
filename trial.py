from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, ensemble, linear_model, svm
from numpy import log, ones, array, zeros, mean, std, repeat
import numpy as np
import scipy.sparse as sp
import re
import csv
from time import time

DIR_PATH = ""

TRAIN_FILE      = DIR_PATH + "train.csv"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"   # This is also used for training, together with TRAIN_FILE
BADWORDS_FILE   = DIR_PATH + "badwords.txt"              # attached with submission  

TEST_FILE       = DIR_PATH + "test.csv"          # set this to the new test file name
PREDICTION_FILE = DIR_PATH + "preds.csv"                 # predictions will be written here 

def readCsv(fname, skipFirst=True, delimiter = ","):
    reader = csv.reader(open(fname),delimiter=delimiter)
    
    rows = []
    count = 1
    for row in reader:
        if not skipFirst or count > 1:      
            rows.append(row)
        count += 1
    return rows

def run(verbose = True):
    t0 = time()

    train_data = readCsv(TRAIN_FILE)
    train2_data = readCsv(TEST_SOL_FILE)
  
    train_data = train_data + train2_data
  #  print(train_data)
    labels  = array([int(x[0]) for x in train_data])
   # print(labels)  
    train  = [x[2] for x in train_data]

    test_data = readCsv(TEST_FILE)
    test_data = [x[2] for x in test_data] 
    
    data = train + test_data
    print(data)
    n = len(data)
    ntrain = len(train)
    



#some code for n grams (use tdifvectorizer)






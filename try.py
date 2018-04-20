from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, ensemble, linear_model, svm
from numpy import log, ones, array, zeros, mean, std, repeat
import numpy as np
import scipy.sparse as sp
import re
import csv
from time import time
import functools
from nltk.util import skipgrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


DIR_PATH = ""

TRAIN_FILE      = DIR_PATH + "train.csv"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"   # This is also used for training, together with TRAIN_FILE
BADWORDS_FILE   = DIR_PATH + "bad_words.txt"              # attached with submission  

TEST_FILE       = DIR_PATH + "test.csv"          # set this to the new test file name
PREDICTION_FILE = DIR_PATH + "preds.csv"                 # predictions will be written here 

def normalize(f , lammatize= False):
    f = [x.lower() for x in f]
    f = [x.replace("\\n"," ") for x in f]        
    f = [x.replace("\\t"," ") for x in f]        
    f = [x.replace("\\xa0"," ") for x in f]
    f = [x.replace("\\xc2"," ") for x in f]

    #f = [x.replace(","," ").replace("."," ").replace(" ", "  ") for x in f]
    #f = [re.subn(" ([a-z]) ","\\1", x)[0] for x in f]  
    #f = [x.replace("  "," ") for x in f]

    f = [x.replace(" u "," you ") for x in f]
    f = [x.replace(" em "," them ") for x in f]
    f = [x.replace(" da "," the ") for x in f]
    f = [x.replace(" yo "," you ") for x in f]
    f = [x.replace(" ur "," you ") for x in f]
    #f = [x.replace(" ur "," your ") for x in f]
    #f = [x.replace(" ur "," you're ") for x in f]
    
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

    #f = [x.replace("outta", "out of") for x in f]

    bwMap = loadBW()
    for key, value in bwMap.items():
        kpad = " " + key + " "
        vpad = " " + value + " "
        f = [x.replace(kpad, vpad) for x in f]
        
    # stemming   
    """
    f = [re.subn("ies( |$)", "y ", x)[0].strip() for x in f]
    #f = [re.subn("([abcdefghijklmnopqrstuvwxyz])s( |$)", "\\1 ", x)[0].strip() for x in f]
    f = [re.subn("s( |$)", " ", x)[0].strip() for x in f]
    f = [re.subn("ing( |$)", " ", x)[0].strip() for x in f]
    f = [x.replace("tard ", " ") for x in f]
        
    f = [re.subn(" [*$%&#@][*$%&#@]+"," xexp ", x)[0].strip() for x in f]
    f = [re.subn(" [0-9]+ "," DD ", x)[0].strip() for x in f]
    f = [re.subn("<\S*>","", x)[0].strip() for x in f]    
    """
    tokenized_sents = [f(i) for i in example]
    if not lammatize:
        stemmer = PorterStemmer()
        for i in range (0, len(tokenized_sents)):
            for j in range (0,len(tokenized_sents[i])):
                tokenized_sents[i][j] = stemmer.stem(tokenized_sents[i][j])
    else:
        lemmatizer = WordNetLemmatizer()
        for i in range (0, len(tokenized_sents)):
            for j in range (0,len(tokenized_sents[i])):
                tokenized_sents[i][j] = lammatizer.lemmatize(tokenized_sents[i][j])    
    for i in tokenized_sents:
        f[i] = " ".join(tokenized_sents[i])
    return f


def ngrams(data, labels, ntrain, min_ngrams=1, max_ngrams=1, no_of_features=500, binary = False, do_normalization = False, stopwords = False, verbose = True, analyzer_char = False):
    f = data
    if do_normalization:
        f = normalize(f)
    
    ftrain = f[:ntrain]
    ftest  = f[ntrain:]
    y_train = labels[:ntrain]
    
    t0 = time()
    analyzer_type = 'word'
    if analyzer_char:
        analyzer_type = 'char'
        
    if binary:
        vectorizer = CountVectorizer(ngram_range = (min_n , max_n), binary =True)
    elif stopwords:
        vectorizer = TfidfVectorizer(ngram_range = (min_n , max_n),stop_words='english',analyzer=analyzer_type,sublinear_tf=True)
    else:
        vectorizer = TfidfVectorizer(ngram_range = (min_n , max_n),sublinear_tf=True,analyzer=analyzer_type)

    if verbose:
        print ("extracting ngrams... where n is [%d,%d]" % (mn,mx))
    
    X_train = vectorizer.fit_transform(ftrain)
    X_test = vectorizer.transform(ftest)
    
    if verbose:
        print ("done in %fs" % (time() - t0), X_train.shape, X_test.shape)

    y = array(y_train)    
    
    numFts = nm
    if numFts < X_train.shape[1]:
        t0 = time()
        ch2 = SelectKBest(chi2, k=numFts)
        X_train = ch2.fit_transform(X_train, y)
        X_test = ch2.transform(X_test)
        assert sp.issparse(X_train)        

    if verbose:
        print ("Extracting best features by a chi-squared test.. ", X_train.shape, X_test.shape  )  
    return X_train, y, X_test


def skipGrams(data, labels, ntrain,nm=500,min_ngrams=1, max_ngrams=1, no_of_features=500, do_normalization = False, verbose = True):
    f = data
    if donorm:
        f = normalize(f)
    
    ftrain = f[:ntrain]
    ftest  = f[ntrain:]
    y_train = labels[:ntrain]
    t0 = time()
    skipper = functools.partial(skipgrams, n=2, k=3)
    
    vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer=analyzer_type)
    
    X_train = vectorizer.fit_transform(ftrain)
    X_test = vectorizer.transform(ftest)
    
    if verbose:
        print ("done in %fs" % (time() - t0), X_train.shape, X_test.shape)

    y = array(y_train)    
    
    numFts = nm
    if numFts < X_train.shape[1]:
        t0 = time()
        ch2 = SelectKBest(chi2, k=numFts)
        X_train = ch2.fit_transform(X_train, y)
        X_test = ch2.transform(X_test)
        assert sp.issparse(X_train) 
    if verbose:
        print ("Extracting best features by a chi-squared test.. ", X_train.shape, X_test.shape) 
    return X_train, y, X_test



def specialCases(data, labels, ntrain, verbose = True):
    g = [x.lower().replace("you are"," SSS ").replace("you're"," SSS ").replace(" ur ", " SSS ").split("SSS")[1:] for x in data]

    f = []
    for x in g:
        fts = " "
        x = normalize(x)
        for y in x:
            w = y.strip().replace("?",".").split(".")
            fts = fts + " " + w[0]        
        f.append(fts)
    
    X_trn, y_trn, X_tst = ngrams(f, labels, ntrain, 1, 1, 100, donorm = True, verbose = verbose)
    return X_trn, y_trn, X_tst



def loadBW():
    f = open(BADWORDS_FILE, "r")
    bwMap = dict()
    for line in f:
        sp = line.strip().lower().split(",")
        if len(sp) == 2:
            bwMap[sp[0].strip()] = sp[1].strip()
    return bwMap

    

def readCsv(fname, skipFirst=True, delimiter = ","):
    reader = csv.reader(open(fname),delimiter=delimiter)
    
    rows = []
    count = 1
    for row in reader:
        if not skipFirst or count > 1:      
            rows.append(row)
        count += 1
    return rows

def write_submission(x,filename):
    wtr = open(filename,"w")
    for i in range(len(x)):
        wtr.write(format(x[i],"0.10f"))
        wtr.write("\n")
    wtr.close()

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
"""
    X_train1, y_train, X_test1 = ngrams(data, labels, ntrain, 1, 1, 2000, do_normalization = True, verbose = verbose)
    X_train2, y_train, X_test2 = ngrams(data, labels, ntrain, 2, 2, 4000, do_normalization = True, verbose = verbose)
    X_train3, y_train, X_test3 = ngrams(data, labels, ntrain, 3, 3, 100,  do_normalization = True, verbose = verbose)    
    X_train4, y_train, X_test4 = ngrams(data, labels, ntrain, 4, 4, 1000, do_normalization = True, verbose = verbose, analyzer_char = True)    
    X_train5, y_train, X_test5 = ngrams(data, labels, ntrain, 5, 5, 1000, do_normalization = True, verbose = verbose, analyzer_char = True)    
    X_train6, y_train, X_test6 = ngrams(data, labels, ntrain, 3, 3, 2000, do_normalization = True, verbose = verbose, analyzer_char = True)    

    X_train7, y_train, X_test7 = specialCases(data, labels, ntrain, verbose = verbose)
    X_train8, y_train, X_test8 = skipGrams(data, labels, ntrain, verbose = verbose)

    X_tn = sp.hstack([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8])
    X_tt = sp.hstack([X_test1,  X_test2,  X_test3, X_test4, X_test5, X_test6, X_test7, X_test8])
    
    if verbose:
        print "######## Total time for feature extraction: %fs" % (time() - t0), X_tn.shape, X_tt.shape
    
    predictions = runClassifiers(X_tn, labels, X_tt)
    
    write_submission(predictions, PREDICTION_FILE)    
    print "Predictions written to:", PREDICTION_FILE
"""

run()
#some code for n grams (use tdifvectorizer)






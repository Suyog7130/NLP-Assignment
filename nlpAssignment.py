
#####################################
###          NLP Assignment       ###
#####################################

"""
18th June 2020:
NLP Assignment to classify type of sentences. 
When reading the txt data, since the separator is ',,,'
which is more than 1 char long and is not '\s+', engine='python'
has to be specified.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report as cr
from sklearn import svm


##-- the main function --##
def main ():
    
    #-- get the data --#
    dfTrain = pd.read_csv('LabelledData (1).txt', sep=' ,,, ', header=None, engine='python')
    dfTrain.columns = ['ques', 'type']
    trainData, trainLabels = dfTrain.ques.to_list(), dfTrain.type.to_list()  #--the train data.
    
    """
    #dfTest = pd.read_csv('train_1000.label', sep='\s+', header=None)
    #dfTest = pd.read_csv('LabelledData (1).txt', sep=' ,,, ', header=None, engine='python')
    
    f = open('train_1000.label', 'r', errors='ignore').read().split('\n')
    testData, testLabels, testType = [], [], []
    for line in f[:len(f)-1]:
        short = line[:20]
        tlabel = short.split(':')[0]
        testLabels.append(tlabel)
        ttype = short.split(' ')[0].split(':')[1]
        testType.append(ttype)
        tdata = line[len(tlabel):]
        testData.append(tdata)
    #print(testLabels); print(testType)"""
    
    cut = 500
    trainDataOld, trainLabelsOld = trainData, trainLabels
    trainData = trainDataOld[:cut]  #int(len(trainData)/2)
    trainLabels = trainLabelsOld[:cut]
    testData = trainDataOld[cut:]
    testLabels = trainLabelsOld[cut:]
    
    #-- vectorizing and training --#
    vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
    trainVectors = vectorizer.fit_transform(trainData)
    testVectors = vectorizer.transform(testData)
    
    #-- performing the classification --#
    model = svm.SVC(kernel='linear')
    model.fit(trainVectors, trainLabels)
    prediction = model.predict(testVectors)
    
    print (cr(testLabels, prediction))

##-- running the main function --##
if __name__=="__main__":
    main()




################# End of Program #######################
########################################################

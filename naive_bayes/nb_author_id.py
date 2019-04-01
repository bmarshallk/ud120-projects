#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### Import libraries ###
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#data-array
features_train

#label-array
labels_train

#Set the classifier and fit it to the training data
classifier = GaussianNB()
t0 = time()
classifier.fit(features_train, labels_train)
print 'Training time: ', round(time()-t0,3), 's'

#Report accuracy by comparing guessed labels vs known-labels on the test set
t1 = time()
labels_predicted = classifier.predict(features_test)
print 'Predicting time: ', round(time()-t1,3), 's'
score = accuracy_score(labels_test, labels_predicted, normalize=True)
print 'The accuracy is: %s' % score



#########################################################



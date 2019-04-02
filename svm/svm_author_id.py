#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
### your code goes here ###
import collections
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Cutting down the numbers of items to train on (speed up training)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

classifier = SVC(C=10000.0, kernel='rbf')
t0 = time()
classifier.fit(features_train, labels_train)
print 'Classifier time: ', round(time()-t0,3), 's'

t1 = time()
predictions = classifier.predict(features_test)
print 'Prediction time: ', round(time()-t1,3), 's'

score = accuracy_score(labels_test, predictions, normalize=True)
print 'The accuracy is: %s' % score

#How many items are Chris's email:
counter = collections.Counter(predictions)
chris = counter[1]
sarah = counter[0]

print 'The number of emails by Chris is: ', chris
#########################################################



#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Splitting up the data into training and test sets to run modelling on
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Picking our classifier, training it, and making predictions
clf = tree.DecisionTreeClassifier().fit(features_train, labels_train)
pred = clf.predict(features_test)

#Evaluating the training by comparing true values to predictions
accuracy = accuracy_score(labels_test, pred)
print 'Accuracy: ', accuracy
precision = precision_score(labels_test, pred)
print 'Precision ', precision
recall = recall_score(labels_test, pred)
print 'Recall ', recall


#Getting the number of true positives from the predictions and label_test (truth)
counter = 0
for i in range(len(pred)):
	if pred[i] == 1 and labels_test == 1:
		counter += 1
print 'Number of true positives: ', counter





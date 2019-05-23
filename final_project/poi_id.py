#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

###Preface: Some information about the dataset
#print('Number of individuals: ', len(data_dict))
#print('Features for each individual: ', data_dict['SKILLING JEFFREY K'].keys())

### Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
full_features_list = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 'email_address',
 'from_poi_to_this_person']

financial_features = ['salary',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'other',
 'director_fees',
 'deferred_income',
 'long_term_incentive']

email_features = ['to_messages',
 'shared_receipt_with_poi',
 'from_messages',
 'from_this_person_to_poi',
 'email_address',
 'from_poi_to_this_person']

### Task 2: Remove outliers
#Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Making a function that identifies outliers by sigma-clipping (for financial data)
def SigmaAlert(feature):
	alerts = []
	#grabbing the values for a given feature into an array (turning values to int() and ignoring NaNs)
	feat_arr = np.array([int(data_dict[key][feature]) for key in data_dict if data_dict[key][feature] != 'NaN'])
	#computing the std-dev and if a value lies outside 3-sigma do the next part
	for value in feat_arr:
		if value > 3*np.std(feat_arr):
			#if above 3-sigma, search and return key for inspection
			for key in data_dict:
				if value == data_dict[key][feature]:
					alerts.append(key)
	return alerts

#Now going through each financial feature and collecting who comes up in SigmaAlerts
all_alerts = []
for feature in financial_features:
	alerts = SigmaAlert(feature)
	print '%s alerts: %s' % (feature,alerts)
	all_alerts.append(alerts)

	#for closer inspection
	for name in alerts:
		print '%s triggers 3-sigma alert on %s. Full info: \n %s' % (name, feature, data_dict[name])
		raw_input('Enter to continue')

	#this returns a list of lists, so we'll re-write it to a single list and uniquify names
	flat_alerts = np.unique([name for alert in all_alerts for name in alert])

'''
#Cycling through alerts once names unique'd
for name in flat_alerts:
	print '%s triggers 3-sigma alert for financial feature(s). Full info: \n %s' % (name, data_dict[name])
	raw_input('Enter to continue')	
'''


'''
#Points we determined to remove (post-inspection)
cleaned_dict = data_dict
cleaned_dict.pop('TOTAL',0) #removing the TOTAL entry in the original dictionary
cleaned_dict.pop('THE TRAVEL AGENCY IN THE PARK',0) #not sure what this is, the only value in nested dict is 'other'
cleaned_dict['BANNANTINE JAMES M']['salary'] = 'NaN' #salary was '477' probably a typo
cleaned_dict['GRAY RODNEY']['salary'] = 'NaN' #salary was '6615' probably a typo
data = featureFormat(cleaned_dict, features_list, sort_keys = True)

#Inspect selected features
features_list = ['salary','exercised_stock_options']
for point in data:
    feat1 = point[0]
    feat2 = point[1]
    #this 
    if feat1 != 0.0 and feat2 != 0.0:
	plt.scatter(feat1, feat2)

plt.xlabel('%s' % features_list[0])
plt.ylabel('%s' % features_list[1])
plt.show()

###Using Pandas scatter_matrix to investigate correlations in financial data (to inspect further):
#findata = featureFormat(cleaned_dict, financial_features, sort_keys = True)
#df = pd.DataFrame(findata)
#scatter_matrix(df)
#plt.show()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
'''

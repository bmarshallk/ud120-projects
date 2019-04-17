#!/usr/bin/python
def outlierCleaner(predictions, ages_train, net_worths_train):


	"""
	Clean away the 10% of points that have the largest
	residual errors (difference between the prediction
	and the actual net worth).

	Return a list of tuples named cleaned_data where 
	each tuple is of the form (age, net_worth, error).
	"""

	#Going through each entry in the array and calculating the error value (value - mean)
	cleaned_data=[]
	eval_data = []
	for i in range(0,predictions.shape[0]):
		#Loading each value up
		age_val = ages_train[i][0]
		net_worths_val = net_worths_train[i][0]
		pred_val = predictions[i][0]
		#Comparing the actual value to the predicted value. 
		error = pred_val - net_worths_val
		ee = error * error #squaring it to remove any value ambiquity of which is largest
		#Storing the values in a tuple and array
		tup = (age_val, net_worths_val, ee)
		eval_data.append(tup)
	
	#this is sorting (in min->max) the array by the error value located within the tuple
	eval_data.sort(key=lambda tup:tup[2]) 

	#Discarding the worst 10% of the data 
	discards = int(0.1 * predictions.shape[0]) # figuring out what th 10% size is
	cleaned_data = eval_data[0:(len(eval_data)-discards)]

	return cleaned_data


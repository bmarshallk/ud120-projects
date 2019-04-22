counter = 0
for name in enron_data.keys():
	if enron_data['%s' % name]['poi'] == True:
		counter += 1
print counter

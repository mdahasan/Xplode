#!/usr/bin/python3

# import libraries
import pandas as pd 
import numpy as np

# import methods from other files
from util import * 

def process_binary_class_data(X, samples_to_add = None):

	column_header = X.columns.values.tolist()	# getting the names of the features
	class_label = column_header[-1]				# getting class column label
	class_column = X.iloc[:, -1].tolist()

	(compatibility, feature_type_list) = check_compatibility(X)
	
	if(compatibility == 'False'):
		print("String data is not supported.")
		return 0
	
	# Get a basic dataset statistics
	if (samples_to_add is None):
		(samples_to_add, minority_class_label) = dataset_stat(class_column)
	else:
		(_, minority_class_label) = dataset_stat(class_column)

	# extract minority dataframe from the original dataframe
	minority_X = X.loc[X[class_label] == minority_class_label]

	synthetic_sample_list = list()

	print('Building synthetic samples...')
	while(len(synthetic_sample_list) < samples_to_add):

		random_sample = generate_random_sample(minority_X, feature_type_list)

		# check acceptibility of the random sample
		accept = ks_test(random_sample, minority_X)

		if(accept == True):
			random_sample.append(minority_class_label)
			synthetic_sample_list.append(random_sample)

		if(len(synthetic_sample_list) > 0 and len(synthetic_sample_list) % 100 == 0):
			print(str(len(synthetic_sample_list)) + ' samples added')

	# creating a dataframe with synthetic samples
	synthetic_DF = pd.DataFrame(synthetic_sample_list, columns = column_header)
	D = X.append(synthetic_DF, ignore_index = True)

	# convert class label to int if class label is string
	if(isinstance(minority_class_label, str)):
		class_column_values = D.loc[:, class_label]
		new_class_label = change_class_label(class_column_values)
		D.drop(class_label, axis = 1, inplace = True)
		D['class'] = pd.Series(new_class_label).values

	# shuffling dataset
	X_shuffle = shuffle_data(D)

	print("----------")
	print(str(samples_to_add) + " samples added to the minority class")
	print("----------------------------------------")

	return X_shuffle, synthetic_DF

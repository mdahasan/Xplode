#!/usr/bin/python3

# import libraries
import numpy as np 
import pandas as pd 
import copy 

# import methods from other file
from util import *
from binary_class_sampling import *

def process_mutliple_class_data(X):

	column_header = X.columns.values.tolist()	# getting the names of the columns
	class_label = column_header[-1]				# getting the name of the class column
	class_column = X.iloc[:, -1].tolist()

	# get the names of all classes
	unique_classes = list(set(class_column))

	# compute the number of samples to add in each class
	max_count = 0
	min_count = 99999999

	for i in range(len(unique_classes)):
		class_count = class_column.count(unique_classes[i])
		if(class_count > max_count):
			max_count = class_count
		if(class_count < min_count):
			min_count = class_count

	final_sample_size = max_count + min_count		# this will be the final sample size. this size will be divided into all classes

	# print all class labels
	print('There are ' + str(len(unique_classes)) + ' classes in the dataset.')

	class_synthetic_sample_dict = dict()		# dict to store synthetic sample for each class label

	# build one-vs-rest binary class dataset
	for i in range(len(unique_classes)):
		base_class = unique_classes[i]
		base_class_count = class_column.count(base_class)
		print('Base class: ' + str(base_class))

		new_class_column = list()				# temp class column to store ovr class label
		for j in range(len(class_column)):
			if(class_column[j] == base_class):
				new_class_column.append(1)
			else:
				new_class_column.append(0)

		temp_X = copy.deepcopy(X)			# make a copy of the original dataframe
		temp_X.drop(class_label, axis = 1, inplace = True)
		temp_X['class'] = pd.Series(new_class_column).values

		_, synthetic_DF = process_binary_class_data(temp_X, (final_sample_size - base_class_count))

		# change class labels back to original class label
		syn_sample, syn_feat = synthetic_DF.shape
		syn_class_label = [base_class] * syn_sample
		synthetic_DF.drop(synthetic_DF.columns[syn_feat - 1], axis = 1)
		synthetic_DF['class'] = syn_class_label

		class_synthetic_sample_dict[base_class] = synthetic_DF

	# appned the synthetic data to the original dataframe
	D = copy.deepcopy(X)
	for key in class_synthetic_sample_dict:
		D = D.append(class_synthetic_sample_dict[key], ignore_index = True)

	# convert class label to int if class label is string
	if(isinstance(unique_classes[0], str)):
		class_column_values = D.loc[:, class_label]
		new_class_label = change_class_label(class_column_values)
		D.drop(class_label, axis = 1, inplace = True)
		D['class'] = pd.Series(new_class_label).values

	# shuffling dataset
	X_shuffle = shuffle_data(D)

	return X_shuffle
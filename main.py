#!/usr/bin/python3

# import libraries
import pandas as pd 
import numpy as np 
import sys
import os

# import functions from other files
from binary_class_sampling import *
from multiple_class_sampling import *
from classification import *

def main():
	# class_option = sys.argv[1]				# -b for binary class, -m for multiclass
	original_data_file = sys.argv[1]		# read original data file

	# getting the name of the dataset
	dataset_name = original_data_file.split('/')[-1]

	print('Working with ' + str(dataset_name.split('.')[0]) + ' dataset...')
	original_X = pd.read_table(original_data_file, sep = ',', header = 'infer')

	# check whether binary/multiple class dataset
	class_column = original_X.iloc[:, -1].tolist()
	unique_classes = list(set(class_column))

	if(len(unique_classes) == 2):
		D, _ = process_binary_class_data(original_X)
		D.to_csv('resampled_' + str(dataset_name), sep = ',')
		binary_classification_report(D)

	elif(len(unique_classes) > 2):
		D = process_mutliple_class_data(original_X)
		D.to_csv('resampled_' + str(dataset_name), sep = ',')
		multiple_classification_report(D)

	else:
		print('Dataset not well formatted!')

if __name__ == "__main__":
	main()
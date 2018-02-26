#!/usr/bin/python3

import numpy as np 
import operator
import random

from scipy import stats
from sklearn import preprocessing

# in file methods
def generate_random_number(bin_dict):

	random_value_prob_dict = dict()
	for key in bin_dict:
		left_edge = key[0]
		right_edge = key[1]

		random_value_prob_dict[random.uniform(left_edge, right_edge)] = bin_dict[key]

	return random_value_prob_dict	

def check_closet_probability_get_value(prob, random_value_prob_dict):

	closet_prob = 1.
	for key in random_value_prob_dict:
		if(abs(prob - random_value_prob_dict[key]) < closet_prob):
			closet_prob = abs(prob - random_value_prob_dict[key])
			selected_value = key

	return selected_value

def check_compatibility(X):

	# check if the dataset is compatible
	# if columns have string values then not compatible
	# if int or float values, then compatible

	# needs to change, code it to accept categorical values

	samples, columns = X.shape

	compatability = True
	feature_type_list = list()

	for i in range(columns - 1):
		value_list = X.iloc[:, i].tolist()
		all_int = all(isinstance(n, int) for n in value_list)
		all_float = all(isinstance(n, float) for n in value_list)

		if(all_int == True):
			compatability = True
			feature_type_list.append('int')
		elif(all_float == True):
			compatability = True
			feature_type_list.append('float')
		else:
			compatability = False

	return compatability, feature_type_list

def dataset_stat(class_column):

	# first basic statistics of the dataset
	# num of sample
	# info about majority and minortiy class

	class_labels = list(set(class_column))
	class_count_dict = dict()
	for i in range(len(class_labels)):
		class_count_dict[class_labels[i]] = class_column.count(class_labels[i])

	sorted_count = sorted(class_count_dict.items(), key = operator.itemgetter(1))

	print('Total number of sample: ' + str(len(class_column)))
	print('Class label: ' + str(class_labels[0]) + ' and ' + str(class_labels[1]))
	print('Minority class label: ' + str(sorted_count[0][0]))
	print('Minority class sample count: ' + str(sorted_count[0][1]))
	print('Majority class label: ' + str(sorted_count[1][0]))
	print('Majority class sample count: ' + str(sorted_count[1][1]))	

	sample_to_add = sorted_count[1][1] - sorted_count[0][1]

	# returning number of samples to add and the name of the minority class
	return (sample_to_add, sorted_count[0][0])

def generate_random_sample(X, feature_type_list):
	
	# all computations are done on minority class dataframe
	samples, columns = X.shape
	col_names = X.columns.values.tolist()

	random_sample = list()

	for i in range(columns - 1):
		bin_dict = dict()
		gap = 0.001

		feat = np.array(X[col_names[i]].tolist())
		freq, edges = np.histogram(feat, bins = 'fd')
		freq_sum = np.sum(freq)

		# build bin range
		for j in range(len(edges) - 1):
			left_edge = edges[j]
			if(left_edge < 0):
				right_edge = edges[j + 1] + gap
			else:
				right_edge = edges[j + 1] - gap

			edge_pair = (left_edge, right_edge)
			bin_dict[edge_pair] = float(freq[j])/freq_sum

		random_value_prob_dict = generate_random_number(bin_dict)

		# generate 100000 uniformly distributed random numbers
		# between (0.001, 0.5) as probability distribution and
		# assign index with each one of them

		# unique random probabilites [0.001, 0.9, 1000]
		probs = np.random.uniform(0.001, 0.9, 1000)

		# unique random indices for each probabilites
		indices = range(1, 1001)
		random.shuffle(indices)

		# choose a probability
		random_index = random.randint(1, 1000)

		prob_index = indices.index(random_index)
		prob = probs[prob_index]

		random_value = check_closet_probability_get_value(prob, random_value_prob_dict)

		if(feature_type_list[i] == 'int'):
			random_value = int(random_value)
		else:
			random_value = float(random_value)

		random_sample.append(random_value)

	return random_sample

def ks_test(x, D):
	samples, feats = D.shape

	p_val_list = list()
	accepting_p_val_list = list()

	for i in range(samples):
		y = D.iloc[i, :-1].tolist()

		ks_stat, p = stats.ks_2samp(x, y)

		p_val_list.append(p)
		if(p > 0.01):
			accepting_p_val_list.append(p)

	# check proportion of satisfying p-value 
	if(len(accepting_p_val_list)/float(samples) > 0.9):
		return True
	else:
		return False

def change_class_label(y):
	le = preprocessing.LabelEncoder()
	le.fit(y)
	new_y = le.transform(y)

	return new_y 

def shuffle_data(D):
	return D.sample(frac = 1).reset_index(drop = True)


	

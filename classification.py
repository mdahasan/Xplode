#!/usr/bin/python3

import sys
import pandas as pd 
import numpy as np 
import copy

from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn import metrics

from util import *

def cross_validation(X, y):

	# classifiers to use
	# clf = svm.SVC(C = 100, gamma = 0.25)
	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = DecisionTreeClassifier(random_state=0)

	# corss-validation scores
	cross_validation_scores = cross_val_score(clf, X, y, cv = 10, n_jobs = -1)
	# print('10-fold cross-validation scores: ')
	# print(cross_validation_scores)
	# print('Average of 10-fold cross-validation score: ' + str(np.mean(cross_validation_scores)))

	return cross_validation_scores, np.mean(cross_validation_scores)

def initial_process(X):

	feature_names = X.columns.values.tolist()
	y = X.iloc[:, -1].tolist()
	X.drop(feature_names[0], axis = 1, inplace = True)		# removin index columns
	X.drop(feature_names[len(feature_names) - 1], axis = 1, inplace = True)		# removin class column

	X = preprocessing.scale(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

	return X, y, X_train, X_test, y_train, y_test

def binary_classification_report(X):

	X, y, X_train, X_test, y_train, y_test = initial_process(X)

	# classifiers to use
	# clf = svm.SVC(C = 100, gamma = 0.25)

	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = DecisionTreeClassifier(random_state=0)

	y_pred = clf.fit(X_train, y_train).predict(X_test)
	# print('ROC AUC Score: ' + str(metrics.roc_auc_score(y_test, y_pred)))
	# print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))

	cv_score, mean_cv_score = cross_validation(X, y)

	return cv_score, mean_cv_score, metrics.roc_auc_score(y_test, y_pred), metrics.accuracy_score(y_test, y_pred)

def one_Vs_one(X):
	column_header = X.columns.values.tolist()	# getting the names of the columns
	class_label = column_header[-1]				# getting the name of the class label
	class_column = X.iloc[:, -1].tolist()

	# get the name of all classes
	unique_classes = list(set(class_column))

	# all pair-wise class classification results
	ovo_result_dict = dict()
	for i in range(len(unique_classes)):
		class1 = unique_classes[i]
		for j in range(i + 1, len(unique_classes)):
			class2 = unique_classes[j]

			# print(str(class1) + ' ' + str(class2))

			ovo_dataset = X.loc[(X[class_label] == class1) | (X[class_label] == class2)]
			# print(ovo_dataset)

			# change class labels
			class_column_values = ovo_dataset.iloc[:, -1]
			new_class_label = change_class_label(class_column_values)
			ovo_dataset.drop(class_label, axis = 1, inplace = True)
			ovo_dataset[class_label] = pd.Series(new_class_label).values

			cv_score, mean_cv_score, roc_auc, acc = binary_classification_report(ovo_dataset)

			class_pair = (class1, class2)

			ovo_result_dict[class_pair] = [cv_score, mean_cv_score, roc_auc, acc]

	return ovo_result_dict

def one_Vs_rest(X):

	column_header = X.columns.values.tolist()	# getting the names of the columns
	class_label = column_header[-1]				# getting the name of the class column
	class_column = X.iloc[:, -1].tolist()

	# get the names of all classes
	unique_classes = list(set(class_column))

	# one vs rest results
	ovr_result_dict = dict()

	for i in range(len(unique_classes)):
		base_class = unique_classes[i]

		new_class_column = list()
		for j in range(len(class_column)):
			if(class_column[j] == base_class):
				new_class_column.append(1)
			else:
				new_class_column.append(0)

		temp_X = copy.deepcopy(X)
		temp_X.drop(class_label, axis = 1, inplace = True)
		temp_X[class_label] = pd.Series(new_class_column).values

		cv_score, mean_cv_score, roc_auc, acc = binary_classification_report(temp_X)

		ovr_result_dict[base_class] = [cv_score, mean_cv_score, roc_auc, acc]

	return ovr_result_dict

def all_class_classification(X):

	X, y, X_train, X_test, y_train, y_test = initial_process(X)

	# classifiers to use
	# clf = svm.SVC()

	# classifiers to use
	# clf = svm.SVC(C = 100, gamma = 0.25)

	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = DecisionTreeClassifier(random_state=0)

	y_pred = clf.fit(X_train, y_train).predict(X_test)
	print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))

	cv_score, mean_cv_score = cross_validation(X, y)
	print('Cross validation score: ', cv_score)
	print('Mean accuracy of 10-fold CV: ', mean_cv_score)

def multiple_classification_report(X):

	# calling one-vs-one method of classification
	ovo_result_dict = one_Vs_one(X)

	print('One Vs. One results: ')
	ovo_acc_list = list()
	ovo_roc_auc_list = list()
	ovo_mean_cv_score_list = list()

	for key in ovo_result_dict:
		class1 = key[0]
		class2 = key[1]

		ovo_cv_score = ovo_result_dict[key][0]
		ovo_mean_cv_score = ovo_result_dict[key][1]
		ovo_roc_auc = ovo_result_dict[key][2]
		ovo_acc = ovo_result_dict[key][3]

		print('Between ' + str(class1) + ' and ' + str(class2) + ': ')
		print('Mean CV score: ' + str(ovo_mean_cv_score))
		print('ROC AUC: ' + str(ovo_roc_auc))
		print('Accuracy: ' + str(ovo_acc))

		ovo_mean_cv_score_list.append(ovo_mean_cv_score)
		ovo_roc_auc_list.append(ovo_roc_auc)
		ovo_acc_list.append(ovo_acc)

		# print(key, ovo_result_dict[key])
	print('Average mean CV score: ' + str(np.mean(ovo_mean_cv_score_list)))
	print('Average ROC AUC: ' + str(np.mean(ovo_roc_auc_list)))
	print('Average OVO accuracy: ' + str(np.mean(ovo_acc_list)))
	print('---------------------------------')
	

	# # calling one-vs-rest method of classification
	# ovr_result_dict = one_Vs_rest(X)

	# print('One Vs.Rest results: ')
	# ovr_acc_list = list()
	# ovr_roc_auc_list = list()
	# ovr_mean_cv_score_list = list()

	# for key in ovr_result_dict:
	# 	class_name = key

	# 	ovr_cv_score = ovr_result_dict[key][0]
	# 	ovr_mean_cv_score = ovr_result_dict[key][1]
	# 	ovr_roc_auc = ovr_result_dict[key][2]
	# 	ovr_acc = ovr_result_dict[key][3]

	# 	print('Between ' + str(class_name) + ': ')
	# 	print('Mean CV score: ' + str(ovr_mean_cv_score))
	# 	print('ROC AUC: ' + str(ovr_roc_auc))
	# 	print('Accuracy: ' + str(ovr_acc))

	# 	ovr_mean_cv_score_list.append(ovr_mean_cv_score)
	# 	ovr_roc_auc_list.append(ovr_roc_auc)
	# 	ovr_acc_list.append(ovr_acc)

	# 	# print(key, ovo_result_dict[key])
	# print('Average mean CV score: ' + str(np.mean(ovr_mean_cv_score_list)))
	# print('Average ROC AUC: ' + str(np.mean(ovr_roc_auc_list)))
	# print('Average Ovr accuracy: ' + str(np.mean(ovr_acc_list)))
	# print('---------------------------------')

	# calling all class classification 
	print('All class together result: ')
	all_class_classification(X)
	print('---------------------------------')
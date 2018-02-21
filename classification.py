#!/usr/bin/python3

import sys
import pandas as pd 
import numpy as np 

from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

def binary_classification_report(X):

	feature_names = X.columns.values.tolist()
	y = X.iloc[:, -1].tolist()
	X.drop(feature_names[0], axis = 1, inplace = True)		# removin index columns
	X.drop(feature_names[len(feature_names) - 1], axis = 1, inplace = True)		# removin class column

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

	# classifiers to use
	clf = svm.SVC()

	y_pred = clf.fit(X_train, y_train).predict(X_test)
	print('ROC AUC Score: ' + str(metrics.roc_auc_score(y_test, y_pred)))
	print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))

	# corss-validation scores
	cross_validation_scores = cross_val_score(clf, X, y, cv = 10, n_jobs = -1)
	print('10-fold cross-validation scores: ')
	print(cross_validation_scores)
	print('Average of 10-fold cross-validation score: ' + str(np.mean(cross_validation_scores)))

def multiple_classification_report(X):

	feature_names = X.columns.values.tolist()
	y = X.iloc[:, -1].tolist()
	X.drop(feature_names[0], axis = 1, inplace = True)		# removin index columns
	X.drop(feature_names[len(feature_names) - 1], axis = 1, inplace = True)		# removin class column

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

	clf = svm.LinearSVC()
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))

	# corss-validation scores
	cross_validation_scores = cross_val_score(clf, X, y, cv = 10, n_jobs = -1)
	print('10-fold cross-validation scores: ')
	print(cross_validation_scores)
	print('Average of 10-fold cross-validation score: ' + str(np.mean(cross_validation_scores)))

import pandas as pd 
import numpy as np 
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def main():
	datafile = sys.argv[1]

	X = pd.read_table(datafile, sep = ',', header = 'infer')
	y = X['class'].values.tolist()
	X.drop('class', axis = 1, inplace = True)
	X.drop('index', axis = 1, inplace = True)

	X = preprocessing.scale(X)

	print(X)
	print(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

	clf = SVC().fit(X_train, y_train)
	print(clf.score(X_test, y_test))


if __name__ == '__main__':
	main()
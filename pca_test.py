import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import preprocessing

import sys
import copy

def decomposition_PCA(df):

	# normalize data
	data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 

	# PCA
	pca = PCA(n_components = 2)
	T = pca.fit_transform(data_scaled)

	print(T)
	print(T.shape)

	projected_T = pca.inverse_transform(T)

	# print(projected_T)
	X_hat = projected_T

	for i in range(len(X_hat)):
		print(X_hat[i])

# load dataset
# iris = datasets.load_iris()
dataset = sys.argv[1]			# load dataset
X = pd.read_table(dataset, sep = ',', header = 'infer')
dataset_name = dataset.split('/')[-1]

column_header = X.columns.values.tolist()	# getting the names of the columns
class_label = column_header[-1]				# getting the name of the class column
class_column = X.iloc[:, -1].tolist()
unique_classes = list(set(class_column))

for i in range(len(unique_classes)):
	base_class = unique_classes[i]

	new_class_col = list()
	for j in range(len(class_column)):
		if(class_column[j] == base_class):
			new_class_col.append(1)
		else:
			new_class_col.append(0)

	temp_X = copy.deepcopy(X)
	temp_X.drop(class_label, axis = 1, inplace = True)
	temp_X['class'] = pd.Series(new_class_col).values

	print(base_class)
	decomposition_PCA(temp_X)

# # normalize data
# from sklearn import preprocessing
# data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 

# # PCA
# pca = PCA(n_components=2)
# pca.fit_transform(data_scaled)

# # Dump components relations with features:
# print pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])
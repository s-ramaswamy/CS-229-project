'''
Jason Ting, Swaroop Ramaswamy
CS 229 Final Project
Main driver for the project. 
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, linear_model

import wrangle
import features

# Load, clean, and wrangle data
df_training, df_test, df_ID_table = wrangle.load_data()
df_training, df_test = wrangle.clean_data(df_training, df_test)
df_training, df_test = wrangle.rename_data(df_training, df_test)
df_training_all, df_test_all = wrangle.merge_data(df_training, df_test, df_ID_table)
print df_training_all.head()

# feature selection
X, Y = features.quick_and_dirty(df_training[2])

# machine learning aka CS229 
# splits for now-in the future we need to make the test matrix from the data
print X.shape
print Y.shape
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
clf = linear_model.LinearRegression().fit(xtrain, ytrain)

print "Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))

# save results

# magic happens!!!

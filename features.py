'''
Jason Ting, Swaroop Ramaswamy
CS 229 Final Project
Calculate and extract features
https://github.com/theusual/kaggle-yelp-business-rating-prediction/blob/master/features.py
'''

from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np
import pandas as pd
from datetime import datetime

# builds a quick and dirty model for project milestone report
def quick_and_dirty(df):
	X = np.matrix(df.user_review_count)
	ones = np.ones((X.size,1))
	print ones.shape
	print X.shape
	print X[0]
	X = np.concatenate((X.T,ones), 1)
	Y = np.matrix(df.user_average_stars)
	return X, Y.T



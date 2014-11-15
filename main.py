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
#X, Y = features.quick_and_dirty(df_training[2])
#X, Y = features.not_so_quick(df_training[2],df_training[1],df_training[0])
TrainMatrix = df_training[0].merge(df_training[1],on="business_id")
TrainMatrix = TrainMatrix.merge(df_training[2],on="user_id")
TestMatrix = df_test[0].merge(df_training[1],on="business_id")
TestMatrix = TestMatrix.merge(df_training[2],on="user_id")
XTrain, YTrain = features.not_so_quick_train(TrainMatrix)

# I'm assuming df is the giant dataframe that includes all the nan values
df_index = df.index.values.tolist()
business_index = df['bus_stars'].index[df['bus_stars'].apply(np.isnan)]
business_index = [df_index.index(i) for i in business_index]
user_index = df['user_average_stars'].index[df['user_average_stars'].apply(np.isnan)]
user_index = [df_index.index(i) for i in user_index]

# finds the indices depending on what is missing
missing_both_index = list(set(business_index) & set(user_index))
missing_business_index = list(set(user_index) - set(business_index))
missing_user_index = list(set(business_index) - set(user_index))

# check the 2nd argument in the iloc index to see if I got the correct features
missing_both_df = df.iloc[missing_both_index,:].values
missing_business_df = df.iloc[missing_business_index,[0,1,5]].values
missing_user_df = df.iloc[missing_user_index,[2,3,4]].values

#XTest = features.not_so_quick_test(TestMatrix)
# machine learning aka CS229 
# splits for now-in the future we need to make the test matrix from the data
xtrain, xtest, ytrain, ytest = train_test_split(XTrain, YTrain)
clf = linear_model.LinearRegression().fit(xtrain, ytrain)

print "RMSE: %.2f" % np.sqrt(np.mean((clf.predict(xtest) - ytest) ** 2))
# print "Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))

# save results

# magic happens!!!

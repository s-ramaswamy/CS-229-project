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

#Load, clean, and wrangle data
#df_training, df_test, df_ID_table = wrangle.load_data()
#df_training, df_test = wrangle.clean_data(df_training, df_test)
#df_training, df_test = wrangle.rename_data(df_training, df_test)
#df_training_all, df_test_all = wrangle.merge_data(df_training, df_test, df_ID_table)
# print df_training_all.head()

# Load the data once and save the pickle files. Use df_training[0].to_pickle('./trainingreviews.pkl') etc.
#trainingreviews = df_training[0]
#trainingbusiness = df_training[1]
#trainingusers = df_training[2]
#testreviews = df_test[0]
#testusers = df_test[2]
#testbusiness = df_test[1]

# loads the data from the code above from pickle failes
trainingreviews = pd.io.pickle.read_pickle('./trainingreviews.pkl')
trainingbusiness = pd.io.pickle.read_pickle('./trainingbusiness.pkl')
trainingusers = pd.io.pickle.read_pickle('./trainingusers.pkl')
testreviews = pd.io.pickle.read_pickle('./testreviews.pkl')
testbusiness = pd.io.pickle.read_pickle('./testbusiness.pkl')
testusers = pd.io.pickle.read_pickle('./testusers.pkl')

# merges the dataframes together
business = trainingbusiness.append(testbusiness)
users = trainingusers.append(testusers)
TrainMatrix = trainingreviews.merge(business,on="business_id")
TrainMatrix = TrainMatrix.merge(users,on="user_id", how='left')
TestMatrix = testreviews.merge(business,on="business_id", how='left')
TestMatrix = TestMatrix.merge(users,on="user_id", how='left')
XTrain,YTrain = features.not_so_quick_train(TrainMatrix)

# splits the dataframe depending on what is missing
missing_both_df, missing_business_df, missing_user_df, missing_none_df = features.separate_df(TestMatrix)

#XTest = features.not_so_quick_test(TestMatrix)
# machine learning aka CS229 
# splits for now-in the future we need to make the test matrix from the data
xtrain, xtest, ytrain, ytest = train_test_split(XTrain, YTrain)
clf = linear_model.LinearRegression().fit(xtrain, ytrain)

print "RMSE: %.2f" % np.sqrt(np.mean((clf.predict(xtest) - ytest) ** 2))
# print "Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))

# save results

# magic happens!!!

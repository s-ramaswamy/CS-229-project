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
# XTrain, YTrain = features.not_so_quick_train(TrainMatrix)
# XTest = features.not_so_quick_test(TestMatrix, TrainMatrix)

# splits the dataframe depending on what is missing
df_index = TestMatrix.index.values.tolist()
business_index = pd.isnull(TestMatrix['bus_stars']).tolist()
business_index = [i for i, elem in enumerate(business_index) if elem]
user_index = pd.isnull(TestMatrix['user_average_stars']).tolist()
user_index = [i for i, elem in enumerate(user_index) if elem]

# finds the indices depending on what is missing
missing_both_index = list(set(business_index) & set(user_index))
missing_user_index = list(set(user_index) - set(business_index))
missing_business_index = list(set(business_index) - set(user_index))
missing_none = list(set(df_index) - set(business_index) - set(user_index))

missing_both_df = TestMatrix.iloc[missing_both_index,:]
missing_business_df = TestMatrix.iloc[missing_business_index, :]
missing_user_df = TestMatrix.iloc[missing_user_index, :]
missing_none_df = TestMatrix.iloc[missing_none, :]
print missing_business_df.head(2)

# machine learning aka CS229 
# splits for now-in the future we need to make the test matrix from the data
'''
clf = linear_model.LinearRegression().fit(XTrain, YTrain)
results = pd.DataFrame(clf.predict(XTest), index = TestMatrix.review_id.values, columns = ['stars'])
results.to_csv('submission.csv')
'''

# print "RMSE: %.2f" % np.sqrt(np.mean((clf.predict(xtest) - ytest) ** 2))

# save results


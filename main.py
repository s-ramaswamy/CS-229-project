'''
Jason Ting, Swaroop Ramaswamy
CS 229 Final Project
Main driver for the project. 
http://scikit-learn.org/stable/auto_examples/imputation.html
'''

import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, ensemble, svm
from sklearn.decomposition import PCA
import wrangle
import features

print 'Started'

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
trainingbusiness = features.add_categories_franchises(trainingbusiness,trainingbusiness)
testbusiness = features.add_categories_franchises(trainingbusiness,testbusiness)
trainingbusiness = features.addwordratings(trainingbusiness,trainingbusiness)
testbusiness = features.addwordratings(trainingbusiness,testbusiness)
business = trainingbusiness.append(testbusiness)
users = trainingusers.append(testusers)
TrainMatrix = trainingreviews.merge(business,on="business_id")
TrainMatrix = TrainMatrix.merge(users,on="user_id", how='left')
TestMatrix = testreviews.merge(business,on="business_id", how='left')
TestMatrix = TestMatrix.merge(users,on="user_id", how='left')

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
missing_none_index = list(set(df_index) - set(business_index) - set(user_index))

missing_both_df = TestMatrix.iloc[missing_both_index,:]
missing_business_df = TestMatrix.iloc[missing_business_index, :]
missing_user_df = TestMatrix.iloc[missing_user_index, :]
missing_none_df = TestMatrix.iloc[missing_none_index, :]

print 'Making models'

XTrain, YTrain, clf_users, clf_biz, clf_both_user, clf_both_biz = features.not_so_quick_train(TrainMatrix)
XTest = features.not_so_quick_test(TestMatrix, TrainMatrix, missing_both_index, missing_user_index, missing_business_index, clf_users, clf_biz, clf_both_user, clf_both_biz)
# XTrain, YTrain = features.multiple_models_train_features(TrainMatrix)
#pca = PCA(n_components =2)
#pca = pca.fit(XTrain)
#XTrain = pca.transform(XTrain)
'''
XTrain1 = features.missing_both_features(TrainMatrix)
XTrain2 = features.missing_business_features(TrainMatrix)
XTrain3 = features.missing_user_features(TrainMatrix)
XTrain4 = features.missing_none_features(TrainMatrix)
'''
# XTest1 = features.missing_both_features(missing_both_df)
# XTest2 = features.missing_business_features(missing_business_df)
# XTest3 = features.missing_user_features(missing_user_df)
# XTest4 = features.missing_none_features(missing_none_df)
#XTest1,XTest2,XTest3,XTest4 = pca.transform(XTest1),pca.transform(XTest2),pca.transform(XTest3),pca.transform(XTest4)
# machine learning aka CS229 
# clf = linear_model.LinearRegression()
# clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
'''
XTest1 = features.missing_both_features(missing_both_df,TrainMatrix)
XTest2 = features.missing_business_features(missing_business_df,TrainMatrix)
XTest3 = features.missing_user_features(missing_user_df,TrainMatrix)
XTest4 = features.missing_none_features(missing_none_df,TrainMatrix)
'''

print 'Fitting models'

#XTest1,XTest2,XTest3,XTest4 = pca.transform(XTest1),pca.transform(XTest2),pca.transform(XTest3),pca.transform(XTest4)
# machine learning aka CS229 
#clf = linear_model.LinearRegression()
# clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
# clf = linear_model.LassoCV(alpha = [0.01, 0.1, 1.0, 10.0])
clf = linear_model.ElasticNetCV(max_iter = 100000, l1_ratio = 0.99)
#clf = ensemble.RandomForestRegressor(n_estimators = 10)
# clf = svm.SVR() doesn't work????
clf.fit(XTrain, np.squeeze(np.asarray(YTrain)))
'''
print 'Fitting the models and predicting'
clf = linear_model.ElasticNetCV(max_iter = 100000, l1_ratio = 0.99).fit(XTrain1,np.squeeze(np.asarray(YTrain)))
results1 = pd.DataFrame(missing_both_df.review_id.values, columns = ['review_id'])
results1['stars'] = clf.predict(XTest1)
clf = linear_model.ElasticNetCV(max_iter = 100000, l1_ratio = 0.99).fit(XTrain2,np.squeeze(np.asarray(YTrain)))
results2 = pd.DataFrame(missing_business_df.review_id.values, columns = ['review_id'])
results2['stars'] = clf.predict(XTest2)
clf = linear_model.ElasticNetCV(max_iter = 100000, l1_ratio = 0.99).fit(XTrain3,np.squeeze(np.asarray(YTrain)))
results3 = pd.DataFrame(missing_user_df.review_id.values, columns = ['review_id'])
results3['stars'] = clf.predict(XTest3)
clf = linear_model.ElasticNetCV(max_iter = 100000, l1_ratio = 0.99).fit(XTrain4,np.squeeze(np.asa rray(YTrain)))
results4 = pd.DataFrame(missing_none_df.review_id.values, columns = ['review_id'])
results4['stars'] = clf.predict(XTest4)
results = results1.append(results2)
results = results.append(results3)
results = results.append(results4)
cf = pd.read_csv('cf.csv')
results.stars[results.review_id.isin(cf.review_id.values)] = cf.stars
results.stars[results['stars'] < 0] = 0
results.stars[results['stars'] > 5] = 5
results.to_csv('submission.csv', index = False)
'''
# save the results
results = pd.DataFrame(TestMatrix.review_id.values, columns = ['review_id'])
results['stars'] = clf.predict(XTest)
results.stars[results['stars'] < 0] = 0
results.stars[results['stars'] > 5] = 5
results.to_csv('submission.csv', index = False)

# print "RMSE: %.2f" % np.sqrt(np.mean((clf.predict(xtest) - ytest) ** 2))
print 'Finished'  

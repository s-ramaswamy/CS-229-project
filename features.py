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

def get_names_list():
    a = list()
    b = list()
    with open("./Data/names.txt","r") as myfile:
        for line in myfile:
            if(line.split()[1]=="f"):
              a.append(line.split()[0])
            else:  
              b.append(line.split()[0])
    return a,b

def get_gender(names):
    gender = np.empty([names.size])
    female_names, male_names = get_names_list()
    i = 0
    for name in names:
        gender[i] = 0
        try:
            if(name.upper() in female_names):
                gender[i] = 1
            elif(name.upper() in male_names):
                gender[i] = -1
        except:
            pass
        i += 1    
    return gender

# builds a model for the complete dataframe by hanlding missing data
def complete():
    return None

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
  
def not_so_quick(users,business,reviews):
    n_reviews = 1000
    userlist = []
    buslist = []
    features = np.empty([1,5]);
    review_stars_vector = np.empty([1,1]);
    female_names, male_names = get_names_list();
    for i in range(n_reviews):
        user_avg_stars = 3.76
        bus_stars = 3.76
        gender = 0
        user_review_count = 10
        bus_review_count = 10
        user = (users[users.user_id == reviews.user_id.iloc[i]])
        if(not user.empty):
            user_avg_stars = user.iat[0,0]
            user_review_count  = user.iat[0,2]
            if(user.iat[0,1].upper() in female_names):
                gender = 1
            elif(user.iat[0,1].upper() in male_names):
                gender = -1
        bus = (business[business.business_id == reviews.business_id.iloc[i]])
        if(not bus.empty):
            bus_stars = bus.iat[0,10]
            bus_review_count = bus.iat[0,9]
        review_stars = reviews.iloc[i]['rev_stars']
        new_features = [user_avg_stars,user_review_count,bus_stars,bus_review_count,gender]
        features = np.vstack([features,new_features]);
        review_stars_vector = np.vstack([review_stars_vector,review_stars]);
    X = np.matrix(features)
    Y = np.matrix(review_stars_vector)
    return X,Y

# main function to build full training model
def not_so_quick_train(block):
    block = block.replace([np.inf, -np.inf], np.nan)
    block.bus_full_address = [v[-5:] for v in block.bus_full_address.values]
    t = block.groupby('bus_full_address').aggregate(np.mean)
    t.fillna(value=3.67, inplace=True)
    # block.merge(t.bus_stars, right_index = True, how='left')
    block.fillna(value=3, inplace=True)
    review_stars_vector = block.rev_stars.values
    user_name = block.user_name.values
    user_average_stars = block.user_average_stars.values
    gender = get_gender(user_name)
    bus_open = block.bus_open.values
    bus_stars = block.bus_stars.values
    bus_review_count = block.bus_review_count.values
    user_review_count = block.user_review_count.values
    funny = block.funny.values
    cool = block.cool.values
    useful= block.useful.values
    category_average = block.category_average
    features = [user_average_stars,gender,bus_open,bus_stars,bus_review_count,user_review_count,funny,cool,useful,category_average]
    X = np.matrix(features).T
    Y = np.matrix(review_stars_vector).T
    return X, Y

def not_so_quick_test(block, train, both_i, user_i, biz_i):
    '''
    block.bus_stars.fillna(value=train.bus_stars.mean())
    block.user_average_stars.fillna(value=train.user_average_stars.mean())
    '''
    block['bus_stars'][biz_i] = np.random.choice(train.bus_stars.values, size = len(biz_i))
    block['bus_stars'][both_i] = np.random.choice(train.bus_stars.values, size = len(both_i))
    block['user_average_stars'][user_i] = np.random.choice(train.user_average_stars.values, size = len(user_i))
    block['user_average_stars'][both_i] = np.random.choice(train.user_average_stars.values, size = len(both_i))
    block.funny.fillna(value=train.funny.mean(),inplace=True)
    block.cool.fillna(value=train.cool.mean(),inplace=True)
    block.useful.fillna(value=train.useful.mean(),inplace=True)
    block.fillna(value=3,inplace=True)
    user_name = block.user_name.values
    user_average_stars = block.user_average_stars.values
    gender = get_gender(user_name)
    bus_open = block.bus_open.values
    bus_stars = block.bus_stars.values
    bus_review_count = block.bus_review_count.values
    user_review_count = block.user_review_count.valuesfunny = block.funny.values
    funny = block.funny.values
    cool = block.cool.values
    useful = block.useful.values
    category_average = block.category_average
    features = [user_average_stars,gender,bus_open,bus_stars,bus_review_count,user_review_count,funny,cool,useful,category_average]
    X = np.matrix(features).T
    return X

def preprocess_users(users):
 	gender = get_gender(users.user_name.values)
 	cool = np.empty([users.size])
 	useful = np.empty([users.size])
 	funny = np.empty([users.size])
 	for index, user in users.iterrows():
 		cool[index] = user.user_votes['cool']
 		useful[index] = user.user_votes['useful']
 		funny[index] = user.user_votes['funny']

# splits the dataframe depending on what is missing
def separate_df(TestMatrix):
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
    return missing_both_df, missing_business_df, missing_user_df, missing_none_df

def franchise_list(block):
    m = block.bus_name.values
    n = list(m)
    franchises = list(set(n))
    d1 = dict()
    d2 = dict((x,n.count(x)) for x in n)
    d2= sorted(d2.items(), key=lambda x: x[1])
    for franchise in franchises:
    	d1[franchise] = block.bus_stars[block.bus_name == franchise].mean()
    return d1,d2 
  
def category_list(block):
    '''Returns two dictionaries. The first is a dictionary with averages for each category
       in the input dataframe. The second is a dictionary with frequencies of each category'''
    l = block.bus_categories.values.tolist()
    l = [item for sublist in l for item in sublist]
    l = list(set(l))
    d1,d2 = dict(),dict()
    for category in l:
        d1[category] = block.bus_stars[block.bus_categories.map(lambda x: category in x)].mean()
        d2[category] = block.bus_stars[block.bus_categories.map(lambda x: category in x)].count()
    return d1,d2    

def add_categories_franchises(trainblock,testblock):
	d1,d2 = category_list(trainblock)
	testblock['category_average'] = 0
	testblock['n_categories'] =  [len(x) for x in testblock.bus_categories.values.tolist()]
	for category in d1:
		testblock['category_average'][testblock.bus_categories.map(lambda x: category in x)] = testblock['category_average'][testblock.bus_categories.map(lambda x: category in x)].values+(d1[category]/testblock['n_categories'][testblock.bus_categories.map(lambda x: category in x)].values)
	d1,d2 = franchise_list(trainblock)
	testblock['franchise_average'] = 0
	for x in testblock['bus_name']:
		if(x in d1):
			testblock['franchise_average'][testblock['bus_name']==x] = d1[x]
		else:
			testblock['franchise_average'][testblock['bus_name']==x] = trainblock.bus_stars.mean()			 
	testblock['franchise_average'] = [d1[x] for x in testblock['bus_name'].values.tolist()]
	testblock['category_average'][testblock['category_average']==0] = trainblock.bus_stars.mean()
	return testblock    
   



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

        
        
   



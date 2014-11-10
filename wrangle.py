'''
Jason Ting, Swaroop Ramaswamy
CS 229 Final Project
Wrangles the data into a pandas dataframe
https://github.com/theusual/kaggle-yelp-business-rating-prediction/blob/master/munge.py
TODO: more cleaning?
'''

import json
import pandas as pd
import numpy as np

# loads the data from a JSON file to a pandas dataframe
def load_data():
	print 'loading data...'
	data_directory = "Data/"
	training_names = ["yelp_training_set_review.json", 
					"yelp_training_set_business.json",
					"yelp_training_set_user.json",
					"yelp_training_set_checkin.json"]
	final_test_names = ["final_test_set_review.json",
					"final_test_set_business.json",
					"final_test_set_user.json",
					"final_test_set_checkin.json"]
	test_names = ["yelp_test_set_review.json",
					"yelp_test_set_business.json",
					"yelp_test_set_user.json", 
					"yelp_test_set_checkin.json"]

	df_training, df_test, df_final_test = [], [], []
	for file_name in training_names:
		json_data = [json.loads(line) for line in open(data_directory + file_name)]
		df_training.append(pd.DataFrame(json_data))
	for file_name in test_names:
		json_data = [json.loads(line) for line in open(data_directory + file_name)]
		df_test.append(pd.DataFrame(json_data))
	for file_name in final_test_names:
		json_data = [json.loads(line) for line in open(data_directory + file_name)]
		df_final_test.append(pd.DataFrame(json_data))
	dfIdLookupTable = pd.read_csv(data_directory+'IdLookupTable.csv')
	
	print 'finished loading'
	return df_training, df_final_test, dfIdLookupTable

# clean data of bad/missing fields
def clean_data(df_training, df_test):
	print 'cleaning data...'
	# clean any bad data, usually by inserting global averages
	df_training[2][df_training[2].average_stars < 1] = df_training[2].average_stars.mean()
	df_test[2][df_test[2].review_count < 1] = df_training[2].review_count.mean()

    # clean bad characters
	df_training[2]['name'] = [x.encode("utf-8") if type(x) != float else x for x in df_training[2]['name']]
	df_test[2]['name'] = [x.encode("utf-8") if type(x) != float else x for x in df_test[2]['name']]
	df_training[1]['name'] = [x.encode("utf-8") if type(x) != float else x for x in df_training[1]['name']]
	df_test[1]['name'] = [x.encode("utf-8") if type(x) != float else x for x in df_test[1]['name']]

	print 'finished cleaning'
	return df_training, df_test

# combines the data together
def merge_data(dfTrn,dfTest, dfIdLookupTable):
	print 'merging data...'
	dfAll = ['','','','']
	for i in (1,2,3):
		dfAll[i] = dfTrn[i].append(dfTest[i])
	dfTest_Tot_BusStars = dfTest[0].merge(dfTrn[1],how='inner',on='business_id')
	dfTest_Tot_UsrStars = dfTest[0].merge(dfTrn[2],how='inner',on='user_id')
	
	# Create benchmark columns for merging with all other data subets
	global_rev_mean = dfTrn[0].rev_stars.mean()
    
    # Business Mean -- Use business mean if known, use global review mean if not
	dfTest_Benchmark_BusMean = dfTest_Tot_BusStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
	dfTest_Benchmark_BusMean['benchmark_bus_mean'] = dfTest_Benchmark_BusMean.bus_stars.fillna(global_rev_mean)
	dfTest_Benchmark_BusMean = dfTest_Benchmark_BusMean.ix[:,['RecommendationId','benchmark_bus_mean']]
	
	# User Mean -- Use user mean if known, global review mean if not
	dfTest_Benchmark_UsrMean = dfTest_Tot_UsrStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
	dfTest_Benchmark_UsrMean['benchmark_usr_mean'] = dfTest_Benchmark_UsrMean.user_average_stars.fillna(global_rev_mean)
	dfTest_Benchmark_UsrMean = dfTest_Benchmark_UsrMean.ix[:,['RecommendationId','benchmark_usr_mean']]
    
    # Business and User mean -- When both use bus_stars, if neither use global review mean
	dfTest_Benchmark_BusUsrMean = dfTest_Tot_BusStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
	dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'] = dfTest_Benchmark_BusUsrMean.bus_stars
	dfTest_Benchmark_BusUsrMean = dfTest_Benchmark_BusUsrMean.merge(dfTest_Tot_UsrStars,how='left',on=['business_id','user_id'])
	dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'] = dfTest_Benchmark_BusUsrMean[np.isnan(dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'])]['user_average_stars']
	for x in range(0,len(dfTest_Benchmark_BusUsrMean)):
		if dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] > 0:
			pass
		elif dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'][x] > 0:
			dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] = dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'][x]
		else:
			dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] = global_rev_mean
    
    # Business and user mean, when both use greater review count, if neither use global review mean
	dfTest_Benchmark_GrtrRevCountMean = dfTest[0].merge(dfTrn[1],how='inner', on='business_id')
	dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_GrtrRevCountMean.merge(dfTrn[2],how='inner', on='user_id')
	dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_GrtrRevCountMean.merge(dfAll[3],how='left', on='business_id')
	dfTemp = dfTest_Benchmark_GrtrRevCountMean[dfTest_Benchmark_GrtrRevCountMean.bus_review_count >= dfTest_Benchmark_GrtrRevCountMean.user_review_count]
	dfTemp['benchmark_grtr_rev_count_mean'] = dfTest_Benchmark_GrtrRevCountMean.bus_stars
	dfTemp2 = dfTest_Benchmark_GrtrRevCountMean[dfTest_Benchmark_GrtrRevCountMean.bus_review_count < dfTest_Benchmark_GrtrRevCountMean.user_review_count]
	dfTemp2['benchmark_grtr_rev_count_mean'] = dfTest_Benchmark_GrtrRevCountMean.user_average_stars
	dfTemp = dfTemp.append(dfTemp2)
	dfTest_Benchmark_BusUsrMean['benchmark_grtr_rev_count_mean'] = dfTemp['benchmark_grtr_rev_count_mean']
	dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_BusUsrMean
	for x in range(0,len(dfTest_Benchmark_GrtrRevCountMean)):
		if dfTest_Benchmark_GrtrRevCountMean['benchmark_grtr_rev_count_mean'][x] > 0:
			pass
		else:
			dfTest_Benchmark_GrtrRevCountMean['benchmark_grtr_rev_count_mean'][x] = dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x]

    ##Create master test set
	dfTest_Master = dfTest[0].merge(dfAll[1],how='left', on='business_id')
	dfTest_Master = dfTest_Master.merge(dfAll[2],how='left', on='user_id')
	dfTest_Master = dfTest_Master.merge(dfAll[3],how='left', on='business_id')

    ##Create set of reviews in training set that match user IDs in test review set but are not contained in the training user set
	dfTemp = pd.DataFrame(dfTest[0]['user_id'].unique())
	dfTemp.columns = ['user_id']
	dfTest_MissingUsers = dfTemp.merge(dfTrn[2],how='left', on='user_id')#;del dfTest_MissingUsers['business_id']
	dfTest_MissingUsers = dfTest_MissingUsers[np.isnan(dfTest_MissingUsers['user_average_stars'])]
	dfTest_MissingUsers = dfTest_MissingUsers.merge(dfTrn[0],on='user_id',how='inner')
	del dfTest_MissingUsers['user_average_stars'];del dfTest_MissingUsers['user_name'];del dfTest_MissingUsers['user_review_count'];del dfTest_MissingUsers['user_votes']

    ## Create _All data subset - has business references with a star rating AND user references with avg stars
	dfTrn_All = dfTrn[0].merge(dfTrn[1],how='inner', on='business_id')
	dfTrn_All = dfTrn_All.merge(dfTrn[2],how='inner', on='user_id')
	dfTrn_All = dfTrn_All.merge(dfAll[3],how='left', on='business_id')

	dfTest_All = dfTest[0].merge(dfTrn[1],how='inner', on='business_id')
	dfTest_All = dfTest_All.merge(dfTrn[2],how='inner', on='user_id')
	dfTest_All = dfTest_All.merge(dfAll[3],how='left', on='business_id')
	print 'finished merging'
	return dfTrn_All, dfTest_All

#rename all columns for clarity, except the keys
def rename_data(dfTrn,dfTest):
	print 'renaming data...'
	dfTrn[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[0]]
	dfTest[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[0]]
	dfTrn[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[1]]
	dfTest[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[1]]
	dfTrn[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[2]]
	dfTest[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[2]]
	dfTrn[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[3]]
	dfTest[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[3]]
	print 'fininshed renaming'
	return dfTrn, dfTest






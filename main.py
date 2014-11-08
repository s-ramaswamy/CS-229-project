'''
Jason Ting, Swaroop Ramaswamy
CS 229 Final Project
Main driver for the project. 
'''

import pandas as pd

import wrangle
import features

# Load, clean, and wrangle data
df_training, df_test, df_ID_table = wrangle.load_data()
df_training, df_test = wrangle.clean_data(df_training, df_test)
df_training, df_test = wrangle.rename_data(df_training, df_test)
df_training_all, df_test_all = wrangle.merge_data(df_training, df_test, df_ID_table)
print df_training_all.head()

# feature selection

# machine learning 

# save results

# magic happens!!!

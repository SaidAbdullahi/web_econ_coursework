import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import re
import random
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

"""Web ecomonics assignemnt 1
   pCTR with Naive Bayes in SkLearn
"""

def load_data(train='Yes', test='No', validation='No'):
	"""Loads and returns datasets as required
	   Return empty lst for if 'No'
	"""
	if train=='Yes':
		df_train = pd.read_csv('dataset/train.csv', sep=',')
	else:
		df_train = []

	if test=='Yes':
		df_test = pd.read_csv('dataset/test.csv', sep=',')
	else:
		df_test = []

	if validation=='Yes':
		df_validation = pd.read_csv('dataset/validation.csv', sep=',')
	else:
		df_validation = []
	print('Data loaded', len(df_train), len(df_test), len(df_validation))
	return df_train, df_test, df_validation


def le_non_integers(df_data, column_name= 'adexchange', le_old= None):
	"""Label encode column. Used as preprocessing non-integer columns  
	   Returns LE (req for new ecoding/decoing) and new column 
	"""
	if le_old== None:
		le = LabelEncoder()
		le.fit(df_data[column_name].unique())
	else:
		le = le_old 
	column_le = le.transform(df_data[column_name])
	#print(np.unique(column_le))
	#print(column_le.shape)
	return le, np.asarray(column_le)


def build_NB_model(df_train):
	"""Format, label encode data and build NB model for specific columns 
	   Return NB_model
	"""
	# y
	array_y = df_train[['click']].as_matrix()
	array_y = np.reshape(array_y, (-1, 1))

	# x (int features)
	array_x_i = df_train[['weekday', 'hour', 'region', 'city', 'slotwidth', 'slotheight', 'advertiser']].as_matrix()
	# x ('non-int converted-to-int' features)
	adexchange_le, col_adexchange_le= le_non_integers(df_train, 'adexchange')
	slotformat_le, col_slotformat_le= le_non_integers(df_train, 'slotformat')
	slotvisibility_le, col_slotvisibility_le= le_non_integers(df_train, 'slotvisibility')
	useragent_le, col_useragent_le= le_non_integers(df_train, 'useragent')

	array_x_ni = np.column_stack((array_x_i, col_adexchange_le, col_slotformat_le, col_slotvisibility_le, col_useragent_le))

	# Model
	NB_model = GaussianNB()
	NB_model.fit(array_x_ni, array_y)

	return NB_model, adexchange_le, slotformat_le, slotvisibility_le, useragent_le

def pred_NB_model(NB_model, df_test, adexchange_le, slotformat_le, slotvisibility_le, useragent_le):
	"""Uses NB_model to predict probabiolitiy on test set
	   Return predictions (mainly 0s) and probabilities
	"""
	# x (int features)
	array_bid = np.asarray(df_test[['bidid']].as_matrix())
	array_x_i = df_test[['weekday', 'hour', 'region', 'city', 'slotwidth', 'slotheight', 'advertiser']].as_matrix()
	# x ('non-int converted-to-int' features). 
	# By providing a xxxxxxx_le we are NOT creating a new encoder
	adexchange_le, t_col_adexchange_le= le_non_integers(df_test, 'adexchange', adexchange_le)
	slotformat_le, t_col_slotformat_le= le_non_integers(df_test, 'slotformat', slotformat_le)
	slotvisibility_le, t_col_slotvisibility_le= le_non_integers(df_test, 'slotvisibility', slotvisibility_le)
	useragent_le, t_col_useragent_le= le_non_integers(df_test, 'useragent', useragent_le)

	array_x_ni = np.column_stack((array_x_i, t_col_adexchange_le, t_col_slotformat_le, t_col_slotvisibility_le, t_col_useragent_le))

	lst_predict_log_proba = []
	lst_predict = []
	for i in range(0, len(df_test)):
	    bid_name = array_bid[i]
	    lst_predict_log_proba.append(NB_model.predict_log_proba(array_x_ni[i]))
	    lst_predict.append(NB_model.predict(array_x_ni[i]))
	    
	return lst_predict_log_proba, lst_predict

df_train, df_test, df_validation= load_data('Yes', 'Yes')
NB_model, adexchange_le, slotformat_le, slotvisibility_le, useragent_le= build_NB_model(df_train)
lst_predict_log_proba, lst_predict= pred_NB_model(NB_model, df_test, adexchange_le, slotformat_le, slotvisibility_le, useragent_le)

np.save('predict_log_proba', np.asarray(lst_predict_log_proba))
np.save('lst_predict_log_proba', np.asarray(lst_predict_log_proba))

print('Script end')


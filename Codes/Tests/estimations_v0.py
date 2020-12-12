####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import os
import json
import argparse

from datetime import datetime
import time

import progressbar
from time import sleep

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

import utils
from utils import epoch_to_date, get_cat, text_clean, resid_cat, is_velocity

from screening_features import screening_continuous, screening_categorical

from transformations import log_transformation, standard_scale, recreate_missings, impute_missing
from transformations import one_hot_encoding

import validation
from validation import KfoldsCV, bootstrap_estimation

####################################################################################################################################
####################################################################################################################################
#############################################################SETTINGS###############################################################

start_time_all = datetime.now()

# Setting arguments for running the script:
parser = argparse.ArgumentParser(description='.')
parser.add_argument("--export", default='metrics', help='Argument to declare whether to export results. For exporting all outputs, choose "all". For exporting only performance metrics, choose "metrics". For no exports, choose "no".')
parser.add_argument("--stores", default='1098', help='Stores argument.')
parser.add_argument("--comment", default='none', help='Comments on estimation task.')
parser.add_argument("--log", default=True, help='Argument to declare the use of logarithmic transformation (choose between True or False.')
parser.add_argument("--stand", default=True, help='Argument to declare whether to standardize numerical data (choose between True or False.')
parser.add_argument("--screen", default=False, help='Argument to declare whether to screen numerical features (choose between True or False.')
parser.add_argument("--screen_option", default='default', help='Argument to declare which specification to use of screening_continuous class (choose among ["none", "default", "winsorize", "drop_outliers", "collinearity"].')
parser.add_argument("--method", default='logistic_regression', help='Argument to define the learning method to be used ["logistic_regression", "GBM"].')
args = parser.parse_args()

# Extracting inputs from arguments:
export = str.lower(args.export)
S = [int(x) for x in args.stores.split(',')]
comment = args.comment
log_transform = "true" == str.lower(args.log)
standardize = "true" == str.lower(args.stand)
screen = "true" == str.lower(args.screen)
screen_option = str.lower(args.screen_option)
method = args.method

# Default paramaters for when 'screen' is set to False:
best_param = np.NaN
new_p_status = 'none'
new_p = np.NaN
updated_new_p = np.NaN

# Dictionary with arguments of screening_continuous class:
screen_specification = dict(zip(['winsorize', 'drop_outliers', 'collinearity'],
                                [False, False, False]))

# Loop over arguments of screening_continuous class:
for k in screen_specification:
    if (screen_option not in ['none', 'default']) & (k == screen_option):
        screen_specification[k] = True

# Number of bootstrap estimations:
num_iter = {
    "logistic_regression": 1000,
    "GBM": 50,
}

# Grids of hyper-parameters:
params = {
    "logistic_regression": {'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.3, 0.5, 0.75, 1, 3, 10]},
    "GBM": {'subsample': [0.75], 'learning_rate': [0.05], 'max_depth': [1,2,3,4],  'n_estimators': [500]}
}

# Default hyper-parameters:
params_default = {
    "logistic_regression": {'C': 1},
    "GBM": {'subsample': 0.75, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 500}
}

print('\n')
print('---------------------------------------------------------------------------------------------------------')
print('\033[1mDEFINITIONS:\033[0m')

print('\033[1mNumber of stores:\033[0m ' + str(len(S)) + '.')
print('\033[1mStores:\033[0m ' + str(S) + '.')
print('\033[1mEstimation method:\033[0m logistic regression.')
print('\033[1mPerformance metrics:\033[0m ROC-AUC, average precision score, and Brier scores.')
if screen:
	print('\033[1mScreening features based on their variances.\033[0m')
	print('\033[1mSpecification of screening_continuous class:\033[0m {0}.'.format(screen_specification))
print('---------------------------------------------------------------------------------------------------------')
print('\n')

estimation_id = str(int(time.time()))
if export == 'all':
	if method == 'logistic_regression':
		os.makedirs('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets/scores/LR/' + estimation_id)
		os.makedirs('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets/model_outcomes/LR/' + estimation_id)

	elif method == 'GBM':
		os.makedirs('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets/scores/GBM/' + estimation_id)
		os.makedirs('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets/model_outcomes/GBM/' + estimation_id)

# Dataset info and performance metrics:
os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality')

if 'metrics.csv' not in os.listdir('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets'):
	metrics = pd.DataFrame(data=[], columns=	["estimation_id", "comment", "screen", "screen_option", "store_id",
												 "n_orders_train", "n_orders_test", "n_vars", "new_p_status", "new_p", "updated_new_p",
												 "share_feat_miss_train", "share_obs_miss_train",
												 "share_feat_miss_test", "share_obs_miss_test",
												 "first_date_train", "last_date_train", "first_date_test", "last_date_test",
												 "avg_order_amount_train", "avg_order_amount_test", "method", "num_iter", "best_param",
												 "test_avg_roc_auc", "test_std_roc_auc", "test_avg_prec_avg", "test_std_prec_avg",
												 "test_avg_brier", "test_std_brier", "running_time"])
	metrics.to_csv('Datasets/metrics.csv', index=False)
	
else:
	metrics = pd.read_csv('Datasets/metrics.csv')

# Performance metrics of all bootstrap estimations:
if 'performance_metrics.json' not in os.listdir('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/Datasets'):
	performance_metrics = {}
	
	with open('Datasets/performance_metrics.json', 'w') as json_file:
		json.dump(performance_metrics, json_file, indent=2)
	
else:
	with open('Datasets/performance_metrics.json') as json_file:
		performance_metrics = json.load(json_file)

bar = progressbar.ProgressBar(maxval=len(S), widgets=['\033[1mExecution progress:\033[0m', progressbar.Bar('=', '[', ']'), ' ',
													  progressbar.Percentage()])
bar.start()

####################################################################################################################################
####################################################################################################################################
#############################################################DATA IMPORT############################################################

for s in S:
	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTORE ' + str(s) + ' (' + str(S.index(s)+1) + '/' + str(len(S)) + ')\033[0m')
	print('\n')

	start_time_store = datetime.now()

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTAGE OF DATA IMPORT\033[0m')
	print('\n')

	print('********************************************************')
	print('\033[1mTasks:\033[0m')
	print('Importing features and labels, and classifying features.')
	print('********************************************************')
	print('\n')

####################################################################################################################################
# Fraud data:

	os.chdir('/home/matheus_rosso/Arquivo/Features/Datasets/')

	df_train = pd.read_csv('dataset_' + str(s) + '.csv', dtype={'order_id': str})
	df_train.drop_duplicates(['order_id', 'epoch', 'order_amount'], inplace=True)

	# Date variable:
	df_train['date'] = df_train.epoch.apply(epoch_to_date)

	# Train-test split:
	df_train['train_test'] = 'test'
	df_train['train_test'].iloc[:int(df_train.shape[0]/2)] = 'train'

	df_test = df_train[df_train['train_test']=='test']
	df_train = df_train[df_train['train_test']=='train']

	# Accessory variables:
	drop_vars = ['y', 'order_amount', 'store_id', 'order_id', 'status', 'epoch', 'date', 'weight', 'train_test']

	print('\033[1mTime interval for store ' + str(s) + ' (training data):\033[0m ' + '(' +
	      str(df_train.date.min().date()) + ', ' + str(df_train.date.max().date()) + ').')
	print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	print('\n')

	print('\033[1mTime interval for store ' + str(s) + ' (test data):\033[0m ' + '(' +
	      str(df_test.date.min().date()) + ', ' + str(df_test.date.max().date()) + ').')
	print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	print('\n')

	# Assessing missing values (training data):
	num_miss_train = df_train.isnull().sum().sum()
	if num_miss_train:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train.isnull().sum().sum()) + '.')
	    print('\n')

	# Assessing missing values (test data):
	num_miss_test = df_test.isnull().sum().sum()
	if num_miss_test:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test.isnull().sum().sum()) + '.')
	    print('\n')

####################################################################################################################################
# Categorical datasets:

	# categorical_train = pd.read_csv('categorical_features/dataset_' + str(s) + '.csv',
	#                       dtype={'order_id': str, 'store_id': int})
	# categorical_train.drop_duplicates(['order_id', 'epoch', 'order_amount'], inplace=True)

	# categorical_train['date'] = categorical_train.epoch.apply(epoch_to_date)

	# # Train-test split:
	# categorical_test = categorical_train[(categorical_train.date > datetime.strptime('2020-03-30', '%Y-%m-%d'))]
	# categorical_train = categorical_train[(categorical_train.date <= datetime.strptime('2020-03-30', '%Y-%m-%d'))]

	# print('\033[1mShape of categorical_train (training data):\033[0m ' + str(categorical_train.shape) + '.')
	# print('\033[1mNumber of orders (training data):\033[0m ' + str(categorical_train.order_id.nunique()) + '.')
	# print('\n')

	# print('\033[1mShape of categorical_test (test data):\033[0m ' + str(categorical_test.shape) + '.')
	# print('\033[1mNumber of orders (test data):\033[0m ' + str(categorical_test.order_id.nunique()) + '.')
	# print('\n')

	# Treating missing values:
	# print('\033[1mAssessing missing values in categorical data (training data):\033[0m')
	# print(categorical_train.drop(drop_vars, axis=1).isnull().sum().sort_values(ascending=False))

	# print('\033[1mAssessing missing values in categorical data (test data):\033[0m')
	# print(categorical_test.drop(drop_vars, axis=1).isnull().sum().sort_values(ascending=False))

	# # Loop over categorical features:
	# for f in categorical_train.drop(drop_vars, axis=1).columns:
	#     # Training data
	#     categorical_train[f] = categorical_train[f].apply(lambda x: 'NA_VALUE' if pd.isna(x) else x)
	    
	#     # Test data:
	#     categorical_test[f] = categorical_test[f].apply(lambda x: 'NA_VALUE' if pd.isna(x) else x)

	# # Assessing missing values:
	# if categorical_train.isnull().sum().sum() > 0:
	#     print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	#           str(categorical_train.isnull().sum().sum()) + '.')
	#     print('\n')

	# if categorical_test.isnull().sum().sum() > 0:
	#     print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	#           str(categorical_test.isnull().sum().sum()) + '.')
	#     print('\n')

	# Treating text data:
	# na_vars = [c for c in categorical_train.drop(drop_vars, axis=1) if 'NA#' in c]

	# # Loop over categorical features:
	# for f in categorical_train.drop(drop_vars, axis=1).drop(na_vars, axis=1).columns:
	#     # Training data:
	#     categorical_train[f] = categorical_train[f].apply(lambda x: text_clean(str(x)))
	    
	#     # Test data:
	#     categorical_test[f] = categorical_test[f].apply(lambda x: text_clean(str(x)))

	# Merging with fraud data:
	# # Training data:
	# df_train = df_train.merge(categorical_train[[f for f in categorical_train.columns if (f not in drop_vars) |
	#                                              (f == 'order_id')]],
	#                           on='order_id', how='left')

	# print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	# print('\n')

	# # Test data:
	# df_test = df_test.merge(categorical_test[[f for f in categorical_test.columns if (f not in drop_vars) |
	#                                           (f == 'order_id')]],
	#                         on='order_id', how='left')

	# print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	# print('\n')

	# # Assessing missing values (training data):
	# if df_train.isnull().sum().sum() != num_miss_train:
	#     print('\033[1mInconsistent number of overall missings values (training data)!\033[0m')
	#     print('\n')

	# # Assessing missing values (test data):
	# if df_test.isnull().sum().sum() != num_miss_test:
	#     print('\033[1mInconsistent number of overall missings values (test data)!\033[0m')
	#     print('\n')

	# # Dropping pre-defined categorical features:
	# original_cat_vars = get_cat(df_train)
	# c_vars = [c for c in list(df_train.columns) if 'C#' in c]

	# na_vars = ['NA#' + c for c in original_cat_vars if 'NA#' + c in list(df_train.columns)]
	# for c in categorical_train.drop(drop_vars, axis=1).columns:
	#     if 'NA#' + c in df_train.columns:
	#         na_vars.append('NA#' + c)
	# na_vars = list(set(na_vars))

	# df_train = df_train.drop(c_vars, axis=1).drop(na_vars, axis=1)
	# df_test = df_test.drop(c_vars, axis=1).drop(na_vars, axis=1)

	# Training data:
	# Listing categorical features:
	cat_vars = get_cat(df_train)

	# Creating a residual category for categorical features:
	for cat in cat_vars:
	    df_train['C#' + cat + '#RESID'] = resid_cat(df_train, cat)

	# Assessing residual category creation:
	for res in df_train.columns[df_train.columns.str.contains('RESID')]:
	    if str(type(df_train[res].iloc[0])) != "<class 'numpy.int64'>":
	        print('Would be better to check on feature ' + res + '!')

	# Converting a set of dummy variables into a single categorical variable:
	cats_check = []

	for cat in cat_vars:
	    cats = df_train[df_train.columns[(df_train.columns.str.contains('C#' + cat)) |
	                         (df_train.columns.isin(['NA#' + cat]))]].stack()
	    ref = pd.DataFrame(data=pd.Series(pd.Categorical(cats[cats!=0].index.get_level_values(1))),
	                       columns = [cat])
	    ref.index = df_train.index
	    df_train[cat] = ref[cat]
	    cats_check.append(len(cats[cats!=0].index.get_level_values(1)) == df_train.shape[0])

	# Checking for missing values:
	if df_train.isnull().sum().sum() > 0:
	    print('\033[1mNumber of identified missings:\033[0m ' + str(df_train.isnull().sum().sum()) + '.')
	    print('\n')
	    
	# Correcting categories names:
	for cat in cat_vars:
	    df_train[cat] = df_train[cat].apply(lambda x: 'NA' if 'NA#' in x else x.split('#')[2])

	# Dropping dummy variables:
	dummy_vars = [c for c in list(df_train.columns) if ('C#' in c)]
	# print('\033[1mDropped missing value features:\033[0m')
	# for c in cat_vars:
	#     if ('NA#' + c) in list(df_train.columns):
	#         dummy_vars.append('NA#' + c)
	#         print('NA#'+ c)

	df_train.drop(dummy_vars, axis=1, inplace=True)

	# Test data:
	# Listing categorical features:
	cat_vars = get_cat(df_test)

	# Creating a residual category for categorical features:
	for cat in cat_vars:
	    df_test['C#' + cat + '#RESID'] = resid_cat(df_test, cat)

	# Assessing residual category creation:
	for res in df_test.columns[df_test.columns.str.contains('RESID')]:
	    if str(type(df_test[res].iloc[0])) != "<class 'numpy.int64'>":
	        print('Would be better to check on feature ' + res + '!')

	# Converting a set of dummy variables into a single categorical variable:
	cats_check = []

	for cat in cat_vars:
	    cats = df_test[df_test.columns[(df_test.columns.str.contains('C#' + cat)) |
	                         (df_test.columns.isin(['NA#' + cat]))]].stack()
	    ref = pd.DataFrame(data=pd.Series(pd.Categorical(cats[cats!=0].index.get_level_values(1))),
	                       columns = [cat])
	    ref.index = df_test.index
	    df_test[cat] = ref[cat]
	    cats_check.append(len(cats[cats!=0].index.get_level_values(1)) == df_test.shape[0])

	# Checking for missing values:
	if df_test.isnull().sum().sum() > 0:
	    print('\033[1mNumber of identified missings:\033[0m ' + str(df_test.isnull().sum().sum()) + '.')
	    print('\n')
	    
	# Correcting categories names:
	for cat in cat_vars:
	    df_test[cat] = df_test[cat].apply(lambda x: 'NA' if 'NA#' in x else x.split('#')[2])

	# Dropping dummy variables:
	dummy_vars = [c for c in list(df_test.columns) if ('C#' in c)]
	# print('\033[1mDropped missing value features:\033[0m')
	# for c in cat_vars:
	#     if ('NA#' + c) in list(df_test.columns):
	#         dummy_vars.append('NA#' + c)
	#         print('NA#'+ c)

	df_test.drop(dummy_vars, axis=1, inplace=True)

####################################################################################################################################
# Classifying features:

	# Categorical features:
	# cat_vars = list(categorical_train.drop(drop_vars, axis=1).columns)

	# # Features with zero variance:
	# no_variance = [c for c in df_train.drop(drop_vars, axis=1).drop(cat_vars, axis=1) if df_train[c].var()==0]

	# # Dropping features with no variance:
	# if len(no_variance) > 0:
	#     df_train.drop(no_variance, axis=1, inplace=True)
	#     df_test.drop(no_variance, axis=1, inplace=True)

	# Dummy variables indicating missing value status:
	missing_vars = [c for c in list(df_train.drop(drop_vars, axis=1).columns) if ('NA#' in c)]

	# Numerical features:
	cont_vars = [c for c in  list(df_train.drop(drop_vars, axis=1).columns) if is_velocity(c)]

	# Binary features:
	binary_vars = [c for c in list(df_train.drop([c for c in df_train.columns if (c in drop_vars) |
	                                             (c in cat_vars) | (c in missing_vars) | (c in cont_vars)],
	                                             axis=1).columns) if set(df_train[c].unique()) == set([0,1])]

	# Updating the list of numerical features:
	for c in list(df_train.drop(drop_vars, axis=1).columns):
	    if (c not in cat_vars) & (c not in missing_vars) & (c not in cont_vars) & (c not in binary_vars):
	        cont_vars.append(c)

	# Dataframe presenting the frequency of features by class:
	feats_assess = pd.DataFrame(data={
	    'class': ['cat_vars', 'missing_vars', 'binary_vars', 'cont_vars', 'drop_vars'],
	    'frequency': [len(cat_vars), len(missing_vars), len(binary_vars), len(cont_vars), len(drop_vars)]
	})
	print(feats_assess.sort_values('frequency', ascending=False))

	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
####################################################################################################################################
#############################################################SCREENING FEATURES#####################################################

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTAGE OF FEATURES SCREENING\033[0m')
	print('\n')

	print('*******************************************************************************************************')
	print('\033[1mTasks:\033[0m')
	print('Selecting features based on their variance.')
	print('*******************************************************************************************************')
	print('\n')

	# Number of features for screening:
	if screen:
		# Importing datasets information:
	    os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/')

	    data_info = pd.read_csv('Datasets/data_info.csv', dtype={'dataset': int})

	# if screen:
	    high_dimensional = data_info[data_info.obs_feat_ratio <= 1]
	    print('\033[1mNumber of high-dimensional datasets:\033[0m ' + str(len(high_dimensional.dataset.to_list())) + '.')
	    print(high_dimensional.dataset.to_list())
	    print('\n')

	    # Average ratio between number of observations and number of features for standard datasets:
	    k_ref = data_info[data_info.obs_feat_ratio > 1].obs_feat_ratio.mean()

	    # Number of features for screening:
	    high_dimensional['new_p'] = high_dimensional.n_obs_train.apply(lambda x: int(np.floor(x/k_ref)))

	# if screen:
	    # Number of features to be selected:
	    new_p = high_dimensional[high_dimensional.dataset==s]['new_p'].values[0]
	    print('\033[1mNumber of features to be selected for store {0}\033[0m: {1}.'.format(s, new_p))
	    print('\n')

	# Screening categorical features:
	# if screen:
	    # Create object for one-hot encoding:
	    categorical_transf = one_hot_encoding(categorical_features = cat_vars)

	    # Creating dummies:
	    # categorical_transf.create_dummies(categorical_train = categorical_train,
	    #                                   categorical_test = categorical_test)
	    categorical_transf.create_dummies(categorical_train = df_train[cat_vars],
	                                      categorical_test = df_test[cat_vars])

	    # Selected dummies:
	    dummy_vars = list(categorical_transf.dummies_train.columns)

	# Screening continuous features:
	# if screen:
	    new_p_status = 'pre_defined'
	    # Updating the number of features to be selected (after picking categorical and binary features):
	    na_binary = ['NA#' + c for c in binary_vars if 'NA#' + c in missing_vars]

	    # updated_new_p = new_p - (len(binary_vars) + len(dummy_vars) + len(na_cat) + len(na_binary))
	    updated_new_p = new_p - (len(dummy_vars) + len(binary_vars) + len(na_binary))
	    
	    if updated_new_p < 0:
	        updated_new_p = new_p
	        new_p_status = 'alternative'
	    
	    print('\033[1mUpdated number of continuous features to be selected for store {0}\033[0m: {1}.'.format(s, updated_new_p))
	    print('\n')

	# if screen:
	    # Declaring object for screening of feature:
	    screening_cont = screening_continuous(features = cont_vars, na_features = missing_vars,
	                                          num_continuous_feat = updated_new_p, stat='variance',
	                                          winsorize = screen_specification['winsorize'],
	                                          drop_outliers = screen_specification['drop_outliers'],
	                                          collinearity = screen_specification['collinearity'])

	    # Screening features:
	    screening_cont.select_feat(df_train)

	    # List of features with the highest variance:
	    selected_cont_feat = screening_cont.selected_feat

	# Keeping only selected features:
	# if screen:
	    binary_na = [n for n in missing_vars if n.replace('NA#', '') in binary_vars]

	    # Training data:
	    df_train = df_train.drop([f for f in df_train.columns if (f not in drop_vars) & (f not in selected_cont_feat) &
	                                                             (f not in binary_vars) & (f not in binary_na)],
	                             axis=1)

	    # Test data:
	    df_test = df_test.drop([f for f in df_test.columns if (f not in drop_vars) & (f not in selected_cont_feat) &
	                                                          (f not in binary_vars) & (f not in binary_na)],
	                            axis=1)

	    print('\033[1mShape of df_train for store ' + str(s) + ':\033[0m ' + str(df_train.shape) + '.')
	    print('\033[1mShape of df_test for store ' + str(s) + ':\033[0m ' + str(df_test.shape) + '.')
	    print('\n')

####################################################################################################################################
####################################################################################################################################
#############################################################DATA PRE-PROCESSING####################################################

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTAGE OF DATA PRE-PROCESSING\033[0m')
	print('\n')

	print('*******************************************************************************************************')
	print('\033[1mTasks:\033[0m')
	print('Assessing missing values, transforming numerical and categorical features, and treating missing values.')
	print('*******************************************************************************************************')
	print('\n')

####################################################################################################################################
# Assessing missing values:

	# Recreating missing values:
	missing_vars = [f for f in df_train.columns if 'NA#' in f]

	# Loop over variables with missing values:
	for f in [c for c in missing_vars if c.replace('NA#', '') not in cat_vars]:
	    if f.replace('NA#', '') in df_train.columns:
	        # Training data:
	        df_train[f.replace('NA#', '')] = recreate_missings(df_train[f.replace('NA#', '')], df_train[f])
	        
	        # Test data:
	        df_test[f.replace('NA#', '')] = recreate_missings(df_test[f.replace('NA#', '')], df_test[f])
	    else:
	        df_train.drop([f], axis=1, inplace=True)
	        df_test.drop([f], axis=1, inplace=True)

	# Dropping all variables with missing value status:
	df_train.drop([f for f in df_train.columns if 'NA#' in f], axis=1, inplace=True)
	df_test.drop([f for f in df_test.columns if 'NA#' in f], axis=1, inplace=True)

	# Describing the frequency of missing values:
	# Dataframe with the number of missings by feature (training data):
	missings_dict = df_train.isnull().sum().sort_values(ascending=False).to_dict()

	missings_assess_train = pd.DataFrame(data={
	    'feature': list(missings_dict.keys()),
	    'missings': list(missings_dict.values())
	})

	share_feat_miss_train = sum(missings_assess_train.missings > 0)/len(missings_assess_train)
	share_obs_miss_train = int(missings_assess_train.missings.mean())/len(df_train)

	print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_assess_train.missings > 0)) +
	      ' out of {} features'.format(len(missings_assess_train)) +
	      ' ({}%).'.format(round((sum(missings_assess_train.missings > 0)/len(missings_assess_train))*100, 2)))
	print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_assess_train.missings.mean())) +
	      ' out of {} observations'.format(len(df_train)) +
	      ' ({}%).'.format(round((int(missings_assess_train.missings.mean())/len(df_train))*100,2)))
	print('\n')
	missings_assess_train.index.name = 'training_data'
	print(missings_assess_train.head(10))
	print('\n')

	# Dataframe with the number of missings by feature (test data):
	missings_dict = df_test.isnull().sum().sort_values(ascending=False).to_dict()

	missings_assess_test = pd.DataFrame(data={
	    'feature': list(missings_dict.keys()),
	    'missings': list(missings_dict.values())
	})

	share_feat_miss_test = sum(missings_assess_test.missings > 0)/len(missings_assess_test)
	share_obs_miss_test = int(missings_assess_test.missings.mean())/len(df_test)

	print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_assess_test.missings > 0)) +
	      ' out of {} features'.format(len(missings_assess_test)) +
	      ' ({}%).'.format(round((sum(missings_assess_test.missings > 0)/len(missings_assess_test))*100, 2)))
	print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_assess_test.missings.mean())) +
	      ' out of {} observations'.format(len(df_test)) +
	      ' ({}%).'.format(round((int(missings_assess_test.missings.mean())/len(df_test))*100,2)))
	print('\n')
	missings_assess_test.index.name = 'test_data'
	print(missings_assess_test.head(10))
	print('\n')

####################################################################################################################################
# Transforming numerical data:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mAPPLYING LOGARITHMIC TRANSFORMATION OVER NUMERICAL DATA\033[0m')
	print('\n')
	# Variables that should not be log-transformed:
	if screen:
	    not_log = [c for c in df_train.columns if c not in [f for f in selected_cont_feat if 'NA#' not in f]]
	else:
	    not_log = [c for c in df_train.columns if c not in cont_vars]

	if log_transform:
	    print('\033[1mTraining data:\033[0m')

	    # Assessing missing values (before logarithmic transformation):
	    num_miss_train = df_train.isnull().sum().sum()
	    if num_miss_train > 0:
	        print('\033[1mNumber of overall missings detected (before logarithmic transformation):\033[0m ' +
	              str(num_miss_train) + '.')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_train)
	    df_train = log_transf.log_transformed

	    # Assessing missing values (after logarithmic transformation):
	    num_miss_train_log = df_train.isnull().sum().sum()
	    if num_miss_train_log > 0:
	        print('\033[1mNumber of overall missings detected (after logarithmic transformation):\033[0m ' + 
	              str(num_miss_train_log) + '.')

	    # Checking consistency in the number of missings:
	    if num_miss_train_log != num_miss_train:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	    print('\n')
	    print('\033[1mTest data:\033[0m')

	    # Assessing missing values (before logarithmic transformation):
	    num_miss_test = df_test.isnull().sum().sum()
	    if num_miss_test > 0:
	        print('\033[1mNumber of overall missings detected (before logarithmic transformation):\033[0m ' +
	              str(num_miss_test) + '.')

	    log_transf = log_transformation(not_log=not_log)
	    log_transf.transform(df_test)
	    df_test = log_transf.log_transformed

	    # Assessing missing values (after logarithmic transformation):
	    num_miss_test_log = df_test.isnull().sum().sum()
	    if num_miss_test_log > 0:
	        print('\033[1mNumber of overall missings detected (after logarithmic transformation):\033[0m ' + 
	              str(num_miss_test_log) + '.')

	    # Checking consistency in the number of missings:
	    if num_miss_test_log != num_miss_test:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	else:
	    print('\033[1mNo transformation performed!\033[0m')

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mAPPLYING STANDARD SCALE TRANSFORMATION OVER NUMERICAL DATA\033[0m')
	print('\n')
	# Inputs that should not be standardized:
	if screen:
	    not_stand = [c for c in df_train.columns if c not in ['L#' + f for f in selected_cont_feat if 'NA#' not in f]]
	else:
	    not_stand = [c for c in df_train.columns if c.replace('L#', '') not in cont_vars]

	if standardize:
	    print('\033[1mTraining data:\033[0m')

	    stand_scale = standard_scale(not_stand = not_stand)
	    stand_scale.scale(train = df_train, test = df_test)
	    df_train_scaled = stand_scale.train_scaled
	    print('\033[1mShape of df_train_scaled (after scaling):\033[0m ' + str(df_train_scaled.shape) + '.')

	    # Assessing missing values (after standardizing numerical features):
	    num_miss_train = df_train.isnull().sum().sum()
	    num_miss_train_scaled = df_train_scaled.isnull().sum().sum()
	    if num_miss_train_scaled > 0:
	        print('\033[1mNumber of overall missings:\033[0m ' + str(num_miss_train_scaled) + '.')
	    else:
	        print('\033[1mNo missing values detected (training data)!\033[0m')

	    if num_miss_train_scaled != num_miss_train:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	    print('\n')
	    print('\033[1mTest data:\033[0m')
	    df_test_scaled = stand_scale.test_scaled
	    print('\033[1mShape of df_test_scaled (after scaling):\033[0m ' + str(df_test_scaled.shape) + '.')

	    # Assessing missing values (after standardizing numerical features):
	    num_miss_test = df_test.isnull().sum().sum()
	    num_miss_test_scaled = df_test_scaled.isnull().sum().sum()
	    if num_miss_test_scaled > 0:
	        print('\033[1mNumber of overall missings:\033[0m ' + str(num_miss_test_scaled) + '.')
	    else:
	        print('\033[1mNo missing values detected (test data)!\033[0m')

	    if num_miss_test_scaled != num_miss_test:
	        print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	else:
	    df_train_scaled = df_train.copy()
	    df_test_scaled = df_test.copy()
	    print('\033[1mNo transformation performed!\033[0m')

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Treating missing values:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mTREATING MISSING VALUES\033[0m')
	print('\n')

	print('\033[1mTraining data:\033[0m')
	num_miss_train = df_train_scaled.isnull().sum().sum()
	print('\033[1mNumber of overall missing values detected before treatment:\033[0m ' +
	      str(num_miss_train) + '.')

	# Loop over features:
	for f in df_train_scaled.drop(drop_vars, axis=1):
	    # Checking if there is missing values for a given feature:
	    if df_train_scaled[f].isnull().sum() > 0:
	        check_missing = impute_missing(df_train_scaled[f])
	        df_train_scaled[f] = check_missing['var']
	        df_train_scaled['NA#' + f.replace('L#', '')] = check_missing['missing_var']

	num_miss_train_treat = int(sum([sum(df_train_scaled[f]) for f in df_train_scaled.columns if 'NA#' in f]))
	print('\033[1mNumber of overall missing values detected during treatment:\033[0m ' +
	      str(num_miss_train_treat) + '.')

	if num_miss_train_treat != num_miss_train:
	    print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	if df_train_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train_scaled.isnull().sum().sum()) + '.')

	print('\n')
	print('\033[1mTest data:\033[0m')
	num_miss_test = df_test_scaled.isnull().sum().sum()
	num_miss_test_treat = 0
	print('\033[1mNumber of overall missing values detected before treatment:\033[0m ' + str(num_miss_test) + '.')

	# Loop over features:
	for f in df_test_scaled.drop(drop_vars, axis=1):
	    # Check if there is dummy variable of missing value status for training data:
	    if 'NA#' + f.replace('L#', '') in list(df_train_scaled.columns):
	        check_missing = impute_missing(df_test_scaled[f])
	        df_test_scaled[f] = check_missing['var']
	        df_test_scaled['NA#' + f.replace('L#', '')] = check_missing['missing_var']
	    else:
	        # Checking if there are missings for variables without missings in training data:
	        if df_test_scaled[f].isnull().sum() > 0:
	            num_miss_test_treat += df_test_scaled[f].isnull().sum()
	            df_test_scaled[f].fillna(0, axis=0, inplace=True)

	num_miss_test_treat += int(sum([sum(df_test_scaled[f]) for f in df_test_scaled.columns if 'NA#' in f]))
	print('\033[1mNumber of overall missing values detected during treatment:\033[0m ' +
	      str(num_miss_test_treat) + '.')

	if num_miss_test_treat != num_miss_test:
	    print('\033[1mProblem - Inconsistent number of overall missings!\033[0m')

	if df_test_scaled.isnull().sum().sum() > 0:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test_scaled.isnull().sum().sum()) + '.')

	print('\n')
	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Transforming categorical data:

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mTRANSFORMING CATEGORICAL FEATURES\033[0m')
	print('\n')

	# Creating dummies through one-hot encoding:
	if screen == False:
	    # Create object for one-hot encoding:
	    categorical_transf = one_hot_encoding(categorical_features = cat_vars)

	    # Creating dummies:
	    # categorical_transf.create_dummies(categorical_train = categorical_train,
	    #                                   categorical_test = categorical_test)
	    categorical_transf.create_dummies(categorical_train = df_train[cat_vars],
	                                      categorical_test = df_test[cat_vars])

	# Training data:
	dummies_train = categorical_transf.dummies_train
	dummies_train.index = df_train.index

	# Test data:
	dummies_test = categorical_transf.dummies_test
	dummies_test.index = df_test.index

	# Dropping original categorical features:
	try:
	    df_train_scaled.drop(cat_vars, axis=1, inplace=True)
	    df_test_scaled.drop(cat_vars, axis=1, inplace=True)

	except:
	    pass

	print('\033[1mNumber of categorical features:\033[0m {}.'.format(len(categorical_transf.categorical_features)))
	print('\033[1mNumber of overall selected dummies:\033[0m {}.'.format(dummies_train.shape[1]))
	print('\033[1mShape of dummies_train for store ' + str(s) + ':\033[0m ' +
	      str(dummies_train.shape) + '.')
	print('\033[1mShape of dummies_test for store ' + str(s) + ':\033[0m ' +
	      str(dummies_test.shape) + '.')
	print('\n')

	# Concatenating all features:
	df_train_scaled = pd.concat([df_train_scaled, dummies_train], axis=1)
	df_test_scaled = pd.concat([df_test_scaled, dummies_test], axis=1)

	print('\033[1mShape of df_train_scaled for store ' + str(s) + ':\033[0m ' + str(df_train_scaled.shape) + '.')
	print('\033[1mShape of df_test_scaled for store ' + str(s) + ':\033[0m ' + str(df_test_scaled.shape) + '.')
	print('\n')

	# Assessing missing values (training data):
	num_miss_train = df_train_scaled.isnull().sum().sum() > 0
	if num_miss_train:
	    print('\033[1mProblem - Number of overall missings detected (training data):\033[0m ' +
	          str(df_train_scaled.isnull().sum().sum()) + '.')
	    print('\n')

	# Assessing missing values (test data):
	num_miss_test = df_test_scaled.isnull().sum().sum() > 0
	if num_miss_test:
	    print('\033[1mProblem - Number of overall missings detected (test data):\033[0m ' +
	          str(df_test_scaled.isnull().sum().sum()) + '.')
	    print('\n')

	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
# Datasets structure:

	# Checking consistency of structure between training and test dataframes:
	if len(list(df_train_scaled.columns)) != len(list(df_test_scaled.columns)):
	    print('\033[1mProblem - Inconsistent number of columns between dataframes for training and test data!\033[0m')

	else:
	    consistency_check = 0
	    
	    # Loop over variables:
	    for c in list(df_train_scaled.columns):
	        if list(df_train_scaled.columns).index(c) != list(df_test_scaled.columns).index(c):
	            print('\033[1mProblem - Feature {0} was positioned differently in training and test dataframes!\033[0m'.format(c))
	            consistency_check += 1
	            
	    # Reordering columns of test dataframe:
	    if consistency_check > 0:
	        ordered_columns = list(df_train_scaled.columns)
	        df_test_scaled = df_test_scaled[ordered_columns]

# ####################################################################################################################################
# ####################################################################################################################################
# ###############################################AVERAGING TRAIN-TEST SPLIT DATA MODELING#############################################

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTAGE OF AVERAGING TRAIN-TEST SPLIT DATA MODELING\033[0m')
	print('\n')

	print('***********************************************')
	print('\033[1mTasks:\033[0m')
	print('Estimating test scores and performance metrics.')
	print('\033[1mEstimation method:\033[0m ' + method.replace('_', ' ') + '.')
	print('\033[1mNumber of iterations:\033[0m {0}.'.format(num_iter[method]))
	print('***********************************************')
	print('\n')

	os.chdir('/home/matheus_rosso/Arquivo/Materiais/Codes/high_dimensionality/')

	print('\033[1mSummary (store ' + str(s) + '):\033[0m')

	performance_metrics[str(s) + '_' + str(screen) + '_' + str(screen_option)] = {}
	start_time = datetime.now()

	try:
	    # Declare averaging estimation object:
	    boot_estimations = bootstrap_estimation(cv=True, task='classification', method=method,
	                                            metric='roc_auc', num_folds=3,
	                                            pre_selecting=False, pre_selecting_param=None, random_search=False,
	                                            grid_param=params[method],
	                                            default_param=params_default[method],
	                                            replacement=True, n_iterations=num_iter[method],
	                                            bootstrap_scores=True)

	    # Running averaging estimation:
	    boot_estimations.run(train_inputs=df_train_scaled.drop(drop_vars, axis=1),
	                         train_output=df_train_scaled['y'],
	                         test_inputs=df_test_scaled.drop(drop_vars, axis=1),
	                         test_output=df_test_scaled['y'])

	    # Predicting scores:
	    score_pred = boot_estimations.boot_scores

	    # Calculating performance metrics:
	    test_avg_roc_auc = boot_estimations.boot_stats['roc_auc']['mean']
	    test_std_roc_auc = boot_estimations.boot_stats['roc_auc']['std']
	    test_avg_prec_avg = boot_estimations.boot_stats['prec_avg']['mean']
	    test_std_prec_avg = boot_estimations.boot_stats['prec_avg']['std']
	    test_avg_brier = boot_estimations.boot_stats['brier']['mean']
	    test_std_brier = boot_estimations.boot_stats['brier']['std']

	    # Exporting test scores:
	    if export == 'all':
	        test_scores = df_test_scaled[['order_id', 'epoch', 'order_amount', 'y']]
	        test_scores['test_score'] = score_pred

	        if method == 'logistic_regression':
	        	test_scores.to_json('Datasets/scores/LR/' + estimation_id + '/test_scores_' + str(s) + '.json')

	        elif method == 'GBM':
	        	test_scores.to_json('Datasets/scores/GBM/' + estimation_id + '/test_scores_' + str(s) + '.json')

	    # Exporting model outcomes:
	    if export == 'all':
	        # Most frequent best hyper-parameters:
	        best_param = [p for p in boot_estimations.performance_metrics['best_param'] if
	                      str(p) == boot_estimations.most_freq_param][0]

	        # Creating estimation object:
	        if method == 'logistic_regression':
		        model = LogisticRegression(solver = 'liblinear',
		                                   penalty = 'l1',
		                                   C = best_param['C'],
		                                   warm_start=True)

	        elif method == 'GBM':
		        model = GradientBoostingClassifier(subsample = float(best_param['subsample']),
				                                   learning_rate = float(best_param['learning_rate']),
				                                   max_depth = int(best_param['max_depth']),
				                                   n_estimators = int(best_param['n_estimators']),
				                                   warm_start=True)

	        # Running estimation:
	        model.fit(df_train_scaled.drop(drop_vars, axis=1), df_train_scaled['y'])

	        if method == 'logistic_regression':
		        # Estimated coefficients:
		        betas = list(model.coef_[0])
		        betas.insert(0, model.intercept_[0])

		        # Dataframe with estimated coefficients:
		        features = list(df_train_scaled.drop(drop_vars, axis=1).columns)
		        features.insert(0, 'bias')
		        model_outcomes = pd.DataFrame(data=features, columns=['feature'])
		        model_outcomes['beta'] = betas
		        model_outcomes.to_json('Datasets/model_outcomes/LR/' + estimation_id + '/model_outcomes_' + str(s) + '.json')

	        elif method == 'GBM':
		        # Estimated feature importances:
		        feat_importance = list(model.feature_importances_)

		        # Dataframe with estimated feature importances:
		        features = list(df_train_scaled.drop(drop_vars, axis=1).columns)
		        model_outcomes = pd.DataFrame(data=features, columns=['feature'])
		        model_outcomes['feat_importance'] = feat_importance
		        model_outcomes.to_json('Datasets/model_outcomes/GBM/' + estimation_id + '/model_outcomes_' + str(s) + '.json')

	except Exception as e:
	    test_avg_roc_auc = np.NaN
	    test_std_roc_auc = np.NaN
	    test_avg_prec_avg = np.NaN
	    test_std_prec_avg = np.NaN
	    test_avg_brier = np.NaN
	    test_std_brier = np.NaN

	    if export == 'all':
	    	if method == 'logistic_regression':
		        pd.DataFrame(data=[]).to_json('Datasets/scores/LR/' + estimation_id + '/test_scores_' + str(s) + '.json')
		        pd.DataFrame(data=[]).to_json('Datasets/model_outcomes/LR/' + estimation_id + '/model_outcomes_' + str(s) + '.json')

	    	elif method == 'GBM':
		        pd.DataFrame(data=[]).to_json('Datasets/scores/GBM/' + estimation_id + '/test_scores_' + str(s) + '.json')
		        pd.DataFrame(data=[]).to_json('Datasets/model_outcomes/GBM/' + estimation_id + '/model_outcomes_' + str(s) + '.json')

	    print('**********************************************************')
	    print('\033[1mProblem - Error when executing train-test estimation:\033[0m')
	    print(e)
	    print('**********************************************************')
	    print('\n')

	end_time = datetime.now()

	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

####################################################################################################################################
####################################################################################################################################
######################################################ESTIMATION OUTCOMES###########################################################

	print('---------------------------------------------------------------------------------------------------------')
	print('\033[1mSTAGE OF ESTIMATION OUTCOMES\033[0m')
	print('\n')

	print('***********************************************')
	print('\033[1mTasks:\033[0m')
	print('Collecting performance metrics and additional information about the estimation.')
	print('***********************************************')
	print('\n')

	end_time_store = datetime.now()

	outcomes = {
	    "estimation_id": estimation_id,
	    "comment": comment,
	    "screen": screen,
	    "screen_option": screen_option,
	    "store_id": s,
	    "n_orders_train": int(df_train_scaled.shape[0]),
	    "n_orders_test": int(df_test_scaled.shape[0]),
	    "n_vars": str(df_train_scaled.shape[1]),
	    "new_p_status": new_p_status,
	    "new_p": new_p,
	    "updated_new_p": updated_new_p,
	    "share_feat_miss_train": share_feat_miss_train,
	    "share_obs_miss_train": share_obs_miss_train,
	    "share_feat_miss_test": share_feat_miss_test,
	    "share_obs_miss_test": share_obs_miss_test,
	    "first_date_train": str(str(df_train_scaled.date.min().date())),
	    "last_date_train": str(df_train_scaled.date.max().date()),
	    "first_date_test": str(str(df_test_scaled.date.min().date())),
	    "last_date_test": str(df_test_scaled.date.max().date()),
	    "avg_order_amount_train":df_train_scaled.order_amount.mean(),
	    "avg_order_amount_test":df_test_scaled.order_amount.mean(),
	    "method": method,
	    "num_iter": num_iter[method],
	    "best_param": str(best_param),
	    "test_avg_roc_auc": test_avg_roc_auc,
	    "test_std_roc_auc": test_std_roc_auc,
	    "test_avg_prec_avg": test_avg_prec_avg,
	    "test_std_prec_avg": test_std_prec_avg,
	    "test_avg_brier": test_avg_brier,
	    "test_std_brier": test_std_brier,
	    "running_time": str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes'
	}

	metrics = pd.concat([metrics, pd.DataFrame(data=outcomes, index=[s])], axis=0, sort=False)

	if (export == 'all') | (export == 'metrics'):
	    metrics.to_csv('Datasets/metrics.csv', index=False)

	# Collecting performance metrics of all bootstrap estimations:
	for k in boot_estimations.performance_metrics.keys():
	    if k != 'best_param':
	        performance_metrics[str(s) + '_' + str(screen) + '_' + str(screen_option)][k] = boot_estimations.performance_metrics[k]

	if (export == 'all') | (export == 'metrics'):
		with open('Datasets/performance_metrics.json', 'w') as json_file:
			json.dump(performance_metrics, json_file, indent=2)

	print('---------------------------------------------------------------------------------------------------------')
	print('\n')

	print('\033[1mRunning time for store ' + str(s) + ':\033[0m ' +
	      str(round(((end_time_store - start_time_store).seconds)/60, 2)) + ' minutes.')
	print('Start time: ' + start_time_store.strftime('%Y-%m-%d') + ', ' + start_time_store.strftime('%H:%M:%S'))
	print('End time: ' + end_time_store.strftime('%Y-%m-%d') + ', ' + end_time_store.strftime('%H:%M:%S'))
	print('\n')

	print('------------------------------------------------------------------------------------------')
	print('\n')

	bar.update(S.index(s)+1)
	sleep(0.1)

	# Assessing last estimation:
	with open('last_estimation.json', 'w') as json_file:
	    json.dump({'last_estimation': str(S.index(s)+1) + ' out of ' + str(len(S)),
	               'store_last_estimation': s}, json_file, indent=2)

# Assessing overall running time:
end_time_all = datetime.now()

print('\n')
print('------------------------------------')
print('\033[1mOverall running time:\033[0m ' + str(round(((end_time_all - start_time_all).seconds)/60, 2)) + ' minutes.')
print('Start time: ' + start_time_all.strftime('%Y-%m-%d') + ', ' + start_time_all.strftime('%H:%M:%S'))
print('End time: ' + end_time_all.strftime('%Y-%m-%d') + ', ' + end_time_all.strftime('%H:%M:%S'))
print('\n')

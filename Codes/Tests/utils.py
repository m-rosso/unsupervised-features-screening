####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np

from datetime import datetime
import time

import re
# pip install unidecode
from unidecode import unidecode

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that converts epoch into date:
def epoch_to_date(x):
    str_datetime = time.strftime('%d %b %Y', time.localtime(x/1000))
    dt = datetime.strptime(str_datetime, '%d %b %Y')
    return dt

####################################################################################################################################
# Function that defines categorical features from dummy variables:

def get_cat(df):
    c_cols = list(df.columns[df.columns.str.contains('C#')])
    c_feat = list(set([x.split('#')[1] for x in c_cols]))
    return c_feat

####################################################################################################################################
# Function that returns residual status for a given categorical feature:
def resid_cat(df, cat_feat):
    assess = df[df.columns[(df.columns.str.contains('C#' + cat_feat)) |
                           (df.columns.isin(['NA#' + cat_feat]))]].sum(axis=1)
    
    if sum([1 for x in assess.unique() if x > 1]) > 0:
        return 'Somethind has gone weird with one hot encoding for this feature!'
    else:
        return assess.apply(lambda x: 1 if x==0 else 0)

####################################################################################################################################
# Function for cleaning texts:
def text_clean(text, lower=True):
    if pd.isnull(text):
        return text
    
    else:
        # Removing accent:
        text_cleaned = unidecode(text)

        # Removing extra spaces:
        text_cleaned = re.sub(' +', ' ', text_cleaned)
        
        # Removing spaces before and after text:
        text_cleaned = str.strip(text_cleaned)
        
        # Replacing spaces:
        text_cleaned = text_cleaned.replace(' ', '_')
        
        # Replacing signs:
        for m in '+-!@#$%¨&*()[]{}\\|':
            if m in text_cleaned:
                text_cleaned = text_cleaned.replace(m, '_')

        # Setting text to lower case:
        if lower:
            text_cleaned = text_cleaned.lower()

        return text_cleaned

####################################################################################################################################
# Function that identifies if a given feature name corresponds to a velocity:
def is_velocity(string):
    if ('C#' in string) | ('NA#' in string):
        return False
    
    x1 = string.split('(')
    
    if len(x1) <= 1:
        return False

    x2 = x1[1]       

    if len(x2) <= 1:
        return False
    
    check = 0
    x3 = x2.split(')')[0].split(',')
    
    if len(x3) == 2:
        first_clause = len([1 for d in '0123456789' if d in x3[0]]) == 0
        second_clause = len([1 for d in '0123456789' if d in x3[1]]) > 0
        third_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[0])]) > 0
        fourth_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[1])]) == 0
        
        if first_clause & second_clause & third_clause & fourth_clause:
            check += 1
    
    return check > 0

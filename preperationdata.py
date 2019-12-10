# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

local_testfilename_val = 'test.csv'
local_trainfilename_val = 'train.csv'
local_validfilename_val = 'valid.csv'


global_train_news = pd.read_csv(local_trainfilename_val)
global_test_news = pd.read_csv(local_testfilename_val)
global_valid_news = pd.read_csv(local_validfilename_val)

def plot_data(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='hls')

def filter_data():
    print("Data filtering: checking for null values")
    global_train_news.isnull().sum()
    global_train_news.info()     
    global_test_news.isnull().sum()
    global_test_news.info()
    global_valid_news.isnull().sum()
    global_valid_news.info()

def print_data():
    print("The following data:")
    print(global_train_news.shape)
    print(global_train_news.head(10))
    print(global_test_news.shape)
    print(global_test_news.head(10)) 
    print(global_valid_news.shape)
    print(global_valid_news.head(10))
    


#checking the data retrived in the graph   
print_data()
#calling graph funtion to check that data is not biased
plot_data(global_train_news)
plot_data(global_test_news)
plot_data(global_valid_news)
#performing data filter checking for null values
#print("debug1-reach");
filter_data();














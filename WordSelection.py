#importing necessary libraries

import numpy as np 
import pandas as pd 
import os
import DataPreparation
from pandas import DataFrame
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk

''' 
#First we visualize the data. You can set your working directory and import file by giving location.
df = pd.read_csv('C:\\Users\\MJs-Razer\\Desktop\\Fake News on Social Media\\Dataset\\data.csv') 
df.shape
df.head()

#get the labels 
text = df.Headline
labels = df.Label
df.drop("Label", axis=1)
labels.head()



#Splitting the dataset for testing and training
x_train,x_test,y_train,y_test = train_test_split(text,labels, test_size=0.30, random_state=42)
print(x_train)
'''


#initializing count_vectorizer
count_vectorizer = CountVectorizer(stop_words = 'english', max_df = 0.7)

#fitting the model using count
count_train =  count_vectorizer.fit_transform(DataPreparation.train_file['Statement'].values)

def countvect_stats():
	count_train.shape()
	print(count_vectorizer.vocabulary_)

#Transforming the test set
#count_test = count_vectorizer.transform() 

#initializing tfidf_vectorizer 
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#fitting the model using Tfidf
tfidf_train =  tfidf_vectorizer.fit_transform(DataPreparation.test_file['Statement'].values)

def tfidfvect_stats():
	tfidf_train.shape()
	print(tfidf_vectorizer.vocabulary_)

tagging = nltk.corpus.treebank.tagged_sents()

cutoff = int(0.75 * len(tagging))
training_sentences = DataPreparation.train_file['Statement']

#Transforming the test set
#tfidf_test = tfidf_vectorizer.transform(x_test) 

'''
#PassiveAggressiveClassifier 
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

y_predict = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_predict)
print(f'Accuracy = {round(score*100,2)}%') 

confusion_matrix(y_test,y_predict, labels=[0,1])
'''
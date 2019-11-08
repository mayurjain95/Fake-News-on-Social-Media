#importing necessary libraries

import numpy as np 
import pandas as pd 
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TFidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#First we visualize the data. You can set your working directory and import file by giving location.
df = pd.read_csv('')
df.shape()
df.head()

#get the labels 
labels = df.labels()
labels.head()

#Splitting the dataset for testing and training
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.30, train_size = 0.70, random_state=42)

tfidf_vectorizer = TFidfVectorizer(stop_words = 'english', max_df = 0.7)

tfidf_train =  tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test) 

#PassiveAggressiveClassifier 
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

y_predict = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy = {round(score*100,2)}%') 

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


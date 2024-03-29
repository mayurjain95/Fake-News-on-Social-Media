# -*- coding: utf-8 -*-
"""
This section of the code perform the Count Vectorization and TF-IDF on the extracted data from preperationdata file also implement stop words
@author: mayank,mayur,ayush
"""

import preperationdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

countV = CountVectorizer()
dummyCV_traincount = countV.fit_transform(preperationdata.global_train_news['Statement'].values)
def print_countV_result():
    dummyCV_traincount.shape
    print(countV.vocabulary_)
    print(countV.get_feature_names()[:25])

tfidfV = TfidfTransformer()
dummy_traintfidf = tfidfV.fit_transform(dummyCV_traincount)
tfidf_trsn = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True)

def print_tfidf_result():
    dummy_traintfidf.shape 
    print(dummy_traintfidf.A[:100])


#operatin for the count vectorizer
#performing count vectorization on the data
print(countV)
print(dummyCV_traincount)
#checking the statististics of the data after applying the count vectorizer
print_countV_result()

#operation for the tfiff
#performing tfidf on the Count Vectorizer problem
print(tfidfV)
print(dummy_traintfidf)
#checking the statistics of the data after applying the tfidf
print_tfidf_result();





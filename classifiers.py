import pandas as pd 
import numpy as np 
import DataPreparation
import WordSelection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#Using Naive Bayes Classifier 
print("Entering Pipeline")
nb_pipeline = Pipeline([('NBCV', WordSelection.count_vectorizer),('nb_clf', MultinomialNB())])
nb_pipeline.fit(DataPreparation.train_file['Statement'], DataPreparation.train_file['Label'])
nb_predict = nb_pipeline.predict(DataPreparation.test_file["Statement"]) 
np.mean(nb_predict == DataPreparation.test_file['Label'])
print("Exit Pipeline")

'''
#Using Logistic Regression Classifier
lr_pipeline = Pipeline([('LRCV', WordSelection.count_vectorizer), ('lr_clf', LogisticRegression())])
lr_pipeline.fit(DataPreparation.train_file['Statement'], DataPreparation.train_file['Label'])
lr_predict = lr_pipeline.predict(DataPreparation.test_file["Statement"]) 
np.mean(lr_predict == DataPreparation.test_file['Label'])

#Using SVM Classifier
svm_pipeline = Pipeline([('SVMCV', WordSelection.count_vectorizer), ('svm_clf', svm.LinearSVC())])
svm_pipeline.fit(DataPreparation.train_file['Statement'], DataPreparation.train_file['Label'])
svm_predict = svm_pipeline.predict(DataPreparation.test_file["Statement"]) 
np.mean(svm_predict == DataPreparation.test_file['Label'])

#Using Random Forest Classifier
rf_pipeline = Pipeline([('RFCV', WordSelection.count_vectorizer), ('rf_clf', RandomForestClassifier(n_estimators=200,n_jobs=3))])
rf_pipeline.fit(DataPreparation.train_file['Statement'], DataPreparation.train_file['Label'])
rf_predict = rf_pipeline.predict(DataPreparation.test_file["Statement"]) 
np.mean(rf_predict == DataPreparation.test_file['Label'])
'''
def analyse_confusion_matrix(classifier):
    kf = KFold(n_splits = 5) 
    scores = []
    confusion_mat = np.array([[0,0], [0,0]])

    for train_index, test_index in kf.split(DataPreparation.train_file):
        train_X = DataPreparation.train_file.loc[train_index]['Statement']
        train_y = DataPreparation.train_file.loc[train_index]['Label']

        test_X = DataPreparation.test_file.loc[test_index]['Statement']
        test_y = DataPreparation.test_file.loc[test_index]['Label']
        #print('shape of test y: ',test_y.shape)
        
        classifier.fit(train_X.values, train_y)
        pred = classifier.predict(test_X)
        print("prediction values is" , pred)
        print('shape of pred: ',pred.shape)
        cm = confusion_matrix(test_y,pred)
        confusion_mat += cm
        fscore = f1_score(test_y,pred, average='weighted')
        scores.append(fscore)
    
    print("Score is:" , sum(scores)/len(scores))
    print("Confusion matrix: \n" )
    print(confusion_mat)

analyse_confusion_matrix(nb_pipeline)
#analyse_confusion_matrix(lr_pipeline)
#analyse_confusion_matrix(svm_pipeline)
#analyse_confusion_matrix(rf_pipeline)


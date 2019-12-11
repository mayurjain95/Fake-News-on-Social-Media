# -*- coding: utf-8 -*-
"""
This the main classifier file and only needed file to be called to run entire code.
Here various classifier is applied on CountVectoriser and TF-IDF data to obtaine the best result.
@author: mayank,mayur,ayush
"""

import preperationdata
import selectfetureondata
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression

#implementation for the count vectorizer
naive_bayes_val = Pipeline([
        ('NB',selectfetureondata.countV),
        ('multiNB_clf',MultinomialNB())])

naive_bayes_val.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
naive_bayes_result = naive_bayes_val.predict(preperationdata.global_test_news['Statement'])
np.mean(naive_bayes_result == preperationdata.global_test_news['Label'])


logistic_regression_val = Pipeline([
        ('LogisticRegr',selectfetureondata.countV),
        ('LogisticReg_clf',LogisticRegression())
        ])

logistic_regression_val.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
LogisticReg_result = logistic_regression_val.predict(preperationdata.global_test_news['Statement'])
np.mean(LogisticReg_result == preperationdata.global_test_news['Label'])

supportvector_machine_val = Pipeline([
        ('svm',selectfetureondata.countV),
        ('svm_clf',svm.SVC(kernel='rbf', C = 1.0))
        ])

supportvector_machine_val.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
svm_result = supportvector_machine_val.predict(preperationdata.global_test_news['Statement'])
np.mean(svm_result == preperationdata.global_test_news['Label'])

randforest_val = Pipeline([
        ('random_fores',selectfetureondata.countV),
        ('randforest_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
randforest_val.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
randonforest_result = randforest_val.predict(preperationdata.global_test_news['Statement'])
np.mean(randonforest_result == preperationdata.global_test_news['Label'])

#implementaion using TFIDF

naive_bayes_val1 = Pipeline([
        ('naivebayes_tfidf',selectfetureondata.tfidf_trsn),
        ('naivebayes_cl',MultinomialNB())])

naive_bayes_val1.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
naive_bayes_resutl = naive_bayes_val1.predict(preperationdata.global_test_news['Statement'])
np.mean(naive_bayes_resutl == preperationdata.global_test_news['Label'])

logisticregression_val1 = Pipeline([
        ('LogisticRegr_tfidf',selectfetureondata.tfidf_trsn),
        ('LogisticRegr_clf',LogisticRegression(penalty="l2",C=1))
        ])

logisticregression_val1.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
logisticregression_result1 = logisticregression_val1.predict(preperationdata.global_test_news['Statement'])
np.mean(logisticregression_result1 == preperationdata.global_test_news['Label'])

supportvector_machine_val1 = Pipeline([
        ('svm-tfidf',selectfetureondata.tfidf_trsn),
        ('svm',svm.SVC(kernel='rbf', C = 1.0))
        ])

supportvector_machine_val1.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
supportvectionmachine_result1 = supportvector_machine_val1.predict(preperationdata.global_test_news['Statement'])
np.mean(supportvectionmachine_result1 == preperationdata.global_test_news['Label'])

randomforest_val1 = Pipeline([
        ('rf_tfidf',selectfetureondata.tfidf_trsn),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
randomforest_val1.fit(preperationdata.global_train_news['Statement'],preperationdata.global_train_news['Label'])
randomforest_result1 = randomforest_val1.predict(preperationdata.global_test_news['Statement'])
np.mean(randomforest_result1 == preperationdata.global_test_news['Label'])
score_r = []
def create_confusionmatrix(classifier):
    
    k_fold = KFold(n_splits=5)
    confusion_mat = np.array([[0,0],[0,0]])
    scores = []
    

    for index_trainset, index_testset in k_fold.split(preperationdata.global_train_news):
        traindata_statement = preperationdata.global_train_news.iloc[index_trainset]['Statement'] 
        traindata_label = preperationdata.global_train_news.iloc[index_trainset]['Label']
    
        testdata_statement = preperationdata.global_train_news.iloc[index_testset]['Statement']
        testdata_label = preperationdata.global_train_news.iloc[index_testset]['Label']
        
        classifier.fit(traindata_statement,traindata_label)
        predic = classifier.predict(testdata_statement)
        
        confusion_mat += confusion_matrix(testdata_label,predic)
        score = f1_score(testdata_label,predic)
        scores.append(score)
    
    return (print('Total statements classified:', len(preperationdata.global_train_news)),
    print('Score length', len(scores)),
    print('Score Result:', sum(scores)/len(scores)),
    print('Confusion matrix:'),
    print(confusion_mat))
    score_r.append(sum(scores)/len(scores))

#KFOld on Classifiers geenrated from countV
print("Implementation one")
print("Result for the Naive Bayes Classifier CountV");
create_confusionmatrix(naive_bayes_val)
print("Result for the Logistic Regression Classifier CountV");
create_confusionmatrix(logistic_regression_val)
print("Result for the Support Vector Machine Classifier CountV");
create_confusionmatrix(supportvector_machine_val)
print("Result for the Random Forest Classifier CountV");
create_confusionmatrix(randforest_val)

#KFold on CLassifer geenrated form tfidf
print("Result for the Naive Bayes classifier - tfidf")
create_confusionmatrix(naive_bayes_val1)
print("Result for the logistic regression classifier -tfidf")
create_confusionmatrix(logisticregression_val1)
print("Result for the Support Vector Machine classifier -tfidf")
create_confusionmatrix(supportvector_machine_val1)
print("Result for the Random Forest classifier -tfidf")
create_confusionmatrix(randomforest_val1)

'''

'''
#obtaining final classification report
print("Classification report Naive Bayes");
print(classification_report(preperationdata.global_test_news['Label'], naive_bayes_resutl))
print("Classification report Logistic regression");
print(classification_report(preperationdata.global_test_news['Label'], logisticregression_result1))
print("Classification report Support Vector Machine");
print(classification_report(preperationdata.global_test_news['Label'], supportvectionmachine_result1))
print("Classification report Random Forest");
print(classification_report(preperationdata.global_test_news['Label'], randomforest_result1))

print("**********************************************end*******************************************************")










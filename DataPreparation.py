import os
import numpy as np 
import pandas as pd
import seaborn as sns

file = os.getcwd()
print(file)

train_file = pd.read_csv(file+"\\train.csv", encoding='utf-8')
print(train_file.head(5))
train_file.shape

test_file = pd.read_csv(file+"\\test.csv", encoding='utf-8')
print(test_file.head(5))
test_file.shape

'''
for line in train_file:
        print (line[0:])
'''

#train_file['Label'] = train_file.index
#test_file['Label'] = test_file.index

print(train_file['Label'])

def create_distribution(datafile):
    return sns.countplot(x='Label', data=datafile, palette='hls')

print(create_distribution(train_file))
print(create_distribution(test_file))

def data_check():
    print("Initializing data check")
    train_file.isnull().sum()
    train_file.info()
    test_file.isnull().sum()
    test_file.info()
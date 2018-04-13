# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:40:37 2018

@author: Shiyao Han
"""
from sklearn.utils import shuffle 
from sklearn import preprocessing

import pandas as pd
from collections import Counter

import numpy as np



def load_data(path_to_train, train_rate):  
    df_train = pd.read_csv(path_to_train)
#    df_train = df_train.iloc[:500]
#    result_types = Counter(list(df_train["label"]))
    result_types = Counter(list(df_train["label"]))
    num_classes = len(result_types.keys())
    df_train = shuffle(df_train)
#    
    n_train = int(df_train.shape[0] * train_rate) 
    train_data = []
    train_result = []
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(df_train["label"]))
    for i in range(df_train.shape[0]):
        a = df_train.iloc[i,1]
        a = list(map(int, a.split()))
        train_data.append(a)
        b = df_train.iloc[i,0]
        tt = lb.transform([b]).tolist()[0]
        train_result.append(tt)
    train_data = np.asarray(train_data)   
    
    train_result = np.asarray(train_result)   
    train_data = train_data.astype('float32')
    return train_data[:n_train], train_result[:n_train], train_data[n_train:], train_result[n_train:], num_classes
    


def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:len(X_all)]
    X_test = X_train_test_normed[len(X_all):]
    return X_all, X_test
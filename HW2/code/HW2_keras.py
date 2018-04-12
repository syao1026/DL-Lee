# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:14:29 2018

@author: Shiyao Han
"""
import keras
#from keras.layers import Dropout
from keras import models
import pandas as pd
import numpy as np
#from sklearn.utils import shuffle 

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

def parse_file(data_frame, train_rate):
    train_result = []
    train_data = []
    n_train = int(data_frame.shape[0] * train_rate)
    for i in range(data_frame.shape[0]):
        data_row = list(data_frame.iloc[i,:])
#        data_row = list(data_frame.iloc[i,:])
        data_row[1] = types[data_row[1]]
        data_row[3] = types[data_row[3]]
        data_row[5] = types[data_row[5]]
        data_row[6] = types[data_row[6]]
        data_row[7] = types[data_row[7]]
        data_row[8] = types[data_row[8]]
        data_row[9] = types[data_row[9]]
        data_row[13] = types[data_row[13]]
        data_row[14] = types[data_row[14]] 
        train_data.append(data_row) 
        train_result.append([1, 0] if data_row.pop(14) == 0 else [0, 1])              
#        if -1 not in data_row:
#            train_data.append(data_row) 
#            train_result.append([1, 0] if data_row.pop(14) == 0 else [0, 1])
    train_data = np.asarray(train_data)
    train_result = np.asarray(train_result)
    return train_data[:n_train], train_result[:n_train], train_data[n_train:], train_result[n_train:]

def get_attributions(data_frame):
    types = {' ?': -1}
    n_col = len(data_frame.columns)
    for j in range(n_col):
        col = list(data_frame.iloc[:,j])
        if isinstance(col[0], str):
#            count += 1 
            features = np.unique(col)
            for fea,i in zip(features, range(len(features))):
                if fea in types:
                    continue;    
                else:
                    types.update({fea: i})
    return types

train_data_ori = pd.read_csv("F:\\CV\\DL-Lee\\HW2\\data\\train.csv")
#train_data_ori = shuffle(train_data_ori)
#train_data_ori = train_data_ori.iloc[:5000]
n_row, n_col = train_data_ori.shape
types = get_attributions(train_data_ori)
# choose 80% of the training data as training data, the rest are validation data
train_rate = 0.8
train_data, train_result, test_data, test_result = parse_file(train_data_ori, train_rate)
train_data, test_data = normalize(train_data, test_data)

batch_size = 100
epoch = 3
n_batch = n_row // batch_size
#dropout = 0.5

model = models.Sequential()
model.add(keras.layers.Dense(units=300, activation='relu', input_shape=(14,)))
#model.add(Dropout(dropout))
model.add(keras.layers.Dense(units=300, activation='relu'))
#model.add(Dropout(dropout))
model.add(keras.layers.Dense(units=100, activation='relu'))
#model.add(Dropout(dropout))
model.add(keras.layers.Dense(units=50, activation='relu'))
#model.add(Dropout(dropout))
model.add(keras.layers.Dense(units=30, activation='relu'))
#model.add(Dropout(dropout))
model.add(keras.layers.Dense(units=2, activation='softmax'))

# RMSprop
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

model.fit(train_data, train_result,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1)

print(model.summary())

score = model.evaluate(train_data, train_result, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#print('dropout: ', dropout)
#print('batch_size', batch_size)

          
#model.save('TFKeras.h5')
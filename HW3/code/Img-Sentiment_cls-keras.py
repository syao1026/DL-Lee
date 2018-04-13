# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:06:14 2018

@author: Shiyao Han
"""
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#import tensorflow as tf
from sklearn.utils import shuffle 
from sklearn import preprocessing
import keras
#from keras.datasets import mnist
#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
import vis 

from keras import models
from keras import layers


#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(get_session())



#from keras import backend as K
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Img-Sentiment-Cls-Keras
def load_data(path_to_train, train_rate):  
    df_train = pd.read_csv(path_to_train)
    df_train = df_train.iloc[:500]
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

def visualize_cat(model, cat):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)
    print(conv_cat.shape)
    plt.imshow(conv_cat)


path_to_train = "F:\\CV\\DL-Lee\\HW3\\data\\train.csv"
train_rate = 0.8
img_rows, img_cols = 48, 48
#num_classes = 7
train_data, train_result, test_data, test_result, num_classes = load_data(path_to_train, train_rate)
#train_data, test_data = normalize(train_data, test_data)
train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
batch_size = 100
epochs = 1



model = models.Sequential()
model.add(layers.Conv2D(500, kernel_size=(3, 3),
                 activation='relu',    
                 padding = 'same',
                 input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


model.add(layers.Conv2D(300, (3, 3), padding='same', activation='relu', ))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(80, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(layers.Flatten())

model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
#model.add(Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.fit(train_data, train_result,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_result))
score = model.evaluate(test_data, test_result, verbose=0)

####################### visualize weight of this "layer_name" layer
weights = []
layer_name = 'conv2d_8'
layer_dict = dict([(layer.name, layer) for layer in model.layers])
plot_x, plot_y = 5,5
fig, ax = plt.subplots(plot_x, plot_y, figsize = (8, 8))
fig.suptitle('Input image and %s filters' % (layer_name,))
fig.tight_layout(pad = 0, rect = [0, 0, 1, 1])
layer = layer_dict[layer_name]
weights = layer.get_weights()[0] # list of numpy arrays
#    biases = layer.get_weights()[1]
for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
    ax[x, y].imshow(weights[:,:,:, y + x * 5].reshape(24, 24), interpolation="nearest", cmap = 'gray')
    ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
    
    
    
######################################### show original image
#plot_x, plot_y = 5,5
#rows, cols = 48, 48
#fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
#fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
#for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
#    img = np.array(train_data[x + y * plot_y])
#    ax[x, y].imshow(img.reshape((rows, cols)), cmap = 'gray')
#    ax[x, y].set_title('Input image')


        
#img = img = np.array(train_data[0])
#fig = plt.imshow(img.reshape((48, 48)), cmap = 'gray')
    

#layer_dict = dict([(layer.name, layer) for layer in model.layers])
#print(model.summary())
#layer_name = 'conv2d_2'
#vis.vis_filter(model, layer_name, layer_dict)

#img = np.array(train_data[0]).reshape((1, 48, 48, 1)).astype(np.float64)
#vis.vis_img_in_filter(img, model, layer_name, layer_dict)


#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
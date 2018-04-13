# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:44:16 2018

@author: Shiyao Han
"""
#from sklearn.utils import shuffle 
#from sklearn import preprocessing
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import models
from keras import layers
#from keras.layers.convolutional import ZeroPadding2D, Convolution2D
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#from keras import backend as K

#
#from keras import models
#from keras import layers


class Vgg16:
        
    def __init__(self,train_data,train_result, input_shape, num_classes, batch_size, epochs):
        self.train_data=train_data
        self.train_result=train_result
        self.batch_size=batch_size
        self.epochs=epochs
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        
###### vgg_16_train
    def VGG_16(self):
        model = Sequential()
#        model.add(ZeroPadding2D((1,1),self.input_shape))
#        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D(1,1), self.input_shape)
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(256, 3, 3, activation='relu'))
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(MaxPooling2D((2,2), strides=(2,2)))
#    
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(ZeroPadding2D((1,1)))
#        model.add(Convolution2D(512, 3, 3, activation='relu'))
#        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='softmax'))
 
# https://keras.io/getting-started/sequential-model-guide/
#configure the learning process, which is done via the compile method. It receives three arguments:
 #An optimizer,  A loss function   , A list of metrics   
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
#  Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the  fit function.       
        model.fit(self.train_data, self.train_result,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)
        score = model.evaluate(self.test_data, self.test_result, verbose=0)
        
        return model, score
    
    def net(self):
        model = models.Sequential()
        model.add(Conv2D(50, kernel_size=(3, 3),
                         activation='relu',    
                         padding = 'same',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        
        
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(100, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        
        model.add(Flatten())
        
        model.add(Dense(500, activation='relu'))
        model.add(Dense(200, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        model.fit(self.train_data, self.train_result,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)
#                  validation_data=(test_data, test_result))
#        score = model.evaluate(test_data, test_result, verbose=0)
        return model
    
#def vis_filter(model, layer_name):
#    layer_dict = dict([(layer.name, layer) for layer in model.layers])
#    weights = layer_dict[layer_name].get_weights()
#    filters = []
#    for filter_index in range(weights[0].shape[3]):
#        w = weights[:, :, :, filter_index]
#        filters.append(w)
    
        


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:52:13 2018

@author: Shiyao Han
"""
import tensorflow as tf
import keras 
#import csv
#import random
#import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle 
#from collections import Counter



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
    return train_data[:n_train], train_result[:n_train], train_data[n_train:], train_result[n_train:]
    ############################
#    train_data = []
#    train_result = []
#    test_data = []
#    test_result = []
#    n_row = data_frame.shape[0]
#    result_types = dict(Counter(list(data_frame["income"])))
#    df_train = pd.DataFrame()
#    df_test = pd.DataFrame()
#    n_cls =int( min(result_types.values()) * train_rate)
#    for tp in result_types.keys():
#        cls = data_frame[data_frame['income'].str.contains(tp)]
#        temp_train = pd.DataFrame(cls.iloc[:n_cls,:])
#        df_train = df_train.append(temp_train)
#        temp_test = pd.DataFrame(cls.iloc[n_cls:,:])
#        df_test = df_test.append(temp_test)
#    ######################################
#    # parse feature
#    for i in range(df_train.shape[0]):
#        data_row = list(df_train.iloc[i,:])
##        data_row = list(data_frame.iloc[i,:])
#        data_row[1] = types[data_row[1]]
#        data_row[3] = types[data_row[3]]
#        data_row[5] = types[data_row[5]]
#        data_row[6] = types[data_row[6]]
#        data_row[7] = types[data_row[7]]
#        data_row[8] = types[data_row[8]]
#        data_row[9] = types[data_row[9]]
#        data_row[13] = types[data_row[13]]
#        data_row[14] = types[data_row[14]]        
#        train_result.append([1, 0] if data_row.pop(14) == 0 else [0, 1])
#        train_data.append(data_row)       
#    for i in range(df_test.shape[0]):
#        data_row = list(df_test.iloc[i,:])
##        data_row = list(data_frame.iloc[i,:])
#        data_row[1] = types[data_row[1]]
#        data_row[3] = types[data_row[3]]
#        data_row[5] = types[data_row[5]]
#        data_row[6] = types[data_row[6]]
#        data_row[7] = types[data_row[7]]
#        data_row[8] = types[data_row[8]]
#        data_row[9] = types[data_row[9]]
#        data_row[13] = types[data_row[13]]
#        data_row[14] = types[data_row[14]]        
#        test_result.append([1, 0] if data_row.pop(14) == 0 else [0, 1])
#        test_data.append(data_row)
#################################################3
            
    
#    return train_data, train_result, train_data, train_result

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


def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return initial
#    initializer = tf.contrib.layers.xavier_initializer()
#    W = tf.Variable(initializer(shape))
#    return W  
    
def bias_variable(shape):
    initial = tf.Variable(tf.constant(0.1, shape=shape))
    return initial
#    initializer = tf.contrib.layers.xavier_initializer()
#    b = tf.Variable(initializer(shape))
#    return b

def get_batch(train_data, train_results, batch, batch_size):
    batch_x = train_data[batch:batch_size+batch]
    batch_y = train_results[batch:batch_size+batch]
    return batch_x, batch_y
    
def load_data(path_to_train_file):
    train_data_ori = pd.read_csv(path_to_train_file)
    train_data_ori = shuffle(train_data_ori)
    return train_data_ori

# parse training data
train_data_ori = pd.read_csv("F:\\CV\\DL-Lee\\HW2\\data\\train.csv")
train_data_ori = shuffle(train_data_ori)
#train_data_ori = train_data_ori.iloc[:5000]
n_row, n_col = train_data_ori.shape
types = get_attributions(train_data_ori)
# choose 80% of the training data as training data, the rest are validation data
train_rate = 0.9
train_data, train_result, test_data, test_result = parse_file(train_data_ori, train_rate)
train_data, test_data = normalize(train_data, test_data)

batch_size = 300
n_batch = n_row // batch_size
    
# deep but not wide network
x = tf.placeholder(tf.float32, [None, len(train_data[0])])
y = tf.placeholder(tf.float32, [None, 2])  

W1 = weight_variable([len(train_data[0]), 1024])
B1 = bias_variable([1024])
 
W2 = weight_variable([1024, 800])
B2 = bias_variable([800])

W3 = weight_variable([800, 500]) 
B3 = bias_variable([500])

W4 = weight_variable([500, 300]) 
B4 = bias_variable([300])

W5 = weight_variable([300, 200]) 
B5 = bias_variable([200])

W6 = weight_variable([200, 100]) 
B6 = bias_variable([100])

W7 = weight_variable([100, 2]) 
B7 = bias_variable([2])


Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y5 = tf.nn.relu(tf.matmul(Y4, W5) + B5)
Y6 = tf.nn.relu(tf.matmul(Y5, W6) + B6)
Y7 = tf.matmul(Y6, W7) + B7

keep_prob = tf.placeholder(tf.float32)
prediction = tf.nn.softmax(Y7)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = prediction))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(5):
        for batch in range(n_batch):
            batch_xs, batch_ys = get_batch(train_data, train_result, batch, batch_size)
#            prediction  = sess.run(prediction, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1})
#            print(prediction)
            loss = sess.run(cross_entropy, feed_dict={x:batch_xs, y:batch_ys}) 
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys}) 
            
        test_acc  = sess.run(accuracy, feed_dict={x:test_data, y:test_result})
        print("Iter" + str(epoch) + ", test accuracy: " + str(test_acc) + ", loss: " + str(loss))
 

       
        
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
#    group = parser.add_mutually_exclusive_group()
#    group.add_argument('--train', action='store_true', default=False,
#                        dest='train', help='Input --train to Train')
#    group.add_argument('--infer', action='store_true',default=False,
#                        dest='infer', help='Input --infer to Infer')
#    parser.add_argument('--train_data_path', type=str,
#                        default='feature/X_train', dest='train_data_path',
#                        help='Path to training data')
#    parser.add_argument('--train_label_path', type=str,
#                        default='feature/Y_train', dest='train_label_path',
#                        help='Path to training data\'s label')
#    parser.add_argument('--test_data_path', type=str,
#                        default='feature/X_test', dest='test_data_path',
#                        help='Path to testing data')
#    parser.add_argument('--save_dir', type=str,
#                        default='logistic_params/', dest='save_dir',
#                        help='Path to save the model parameters')
#    parser.add_argument('--output_dir', type=str,
#                        default='logistic_output/', dest='output_dir',
#                        help='Path to save the model parameters')
#    opts = parser.parse_args()
#    main(opts)
#
#    
#
#

imp# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:52:13 2018

@author: Shiyao Han
"""
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle 
from sklearn import preprocessing
#from collections import Counter

def load_data(path_to_train, train_rate):  
    df_train = pd.read_csv(path_to_train)
    df_train = shuffle(df_train)
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
#    train_result = np.asarray(train_result)
    train_data = train_data.astype('float32')    
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
    with tf.device("/gpu:0"):
        initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    return initial
    
def bias_variable(shape):
    with tf.device("/gpu:0"):
        initial = tf.Variable(tf.constant(0.1, shape=shape))
    return initial


def conv2d(x,W):
    # [0] and [3] in strides are 0 by defalt, [1] & [2] are step in x and y direction
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # ksize: 2*2 window, strides[1],[2] step = 2 in x and y direction
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_batch(train_data, train_results, batch, batch_size):
    batch_x = train_data[batch:batch_size+batch]
    batch_y = train_results[batch:batch_size+batch]
    return batch_x, batch_y
    
#def load_data(path_to_train_file):
#    train_data_ori = pd.read_csv(path_to_train_file)
#    train_data_ori = shuffle(train_data_ori)
#    return train_data_ori

# parse training data
path_to_train = "F:\\CV\\DL-Lee\\HW3\\data\\train.csv"
train_rate = 0.8
img_rows, img_cols = 48, 48
train_data, train_result, test_data, test_result = load_data(path_to_train, train_rate)
train_data, test_data = normalize(train_data, test_data)

batch_size = 10
n_batch = train_data.shape[0] // batch_size
    
# deep but not wide network
x = tf.placeholder(tf.float32, [None, len(train_data[0])])
y = tf.placeholder(tf.float32, [None, 7])  

x_image = tf.reshape(x,[-1,img_rows, img_cols,1])
# the origianl image is 48 * 48 (row by row)
W_conv1 = weight_variable([3,3,1,32]) # 5*5windiw, 32 conv kernel on one chaneel
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
 
# 24 * 24 * 32
W_conv2 = weight_variable([3,3,32,64]) # 5*5windiw, 64 conv kernel on 32 chaneel
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 12 * 12 * 64
W_conv3 = weight_variable([3,3,64,48]) # 5*5windiw, 64 conv kernel on 32 chaneel
b_conv3 = bias_variable([48])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# 6 * 6 * 48
W_fc1 = weight_variable([6*6*48,500])
b_fc1 = bias_variable([500])
h_pool3_flat = tf.reshape(h_pool3, [-1,6*6*48])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([500,7])
b_fc2 = bias_variable([7])


#Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
#Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
#Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
#Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
#Y5 = tf.matmul(Y4, W5) + B5

#keep_prob = tf.placeholder(tf.float32)
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = prediction))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#with tf.Session(config=config) as sess:
sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))  
with tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs, batch_ys = get_batch(train_data, train_result, batch, batch_size)
#            prediction  = sess.run(prediction, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1})
#            print(prediction)
            loss = sess.run(cross_entropy, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1}) 
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1}) 
            
        test_acc  = sess.run(accuracy, feed_dict={x:test_data, y:test_result, keep_prob: 1})
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

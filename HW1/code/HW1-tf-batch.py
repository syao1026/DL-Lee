# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:53:02 2018

@author: Shiyao Han
"""


import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import tensorflow as tf
import pandas as pd


######## read data
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('F:\\ML-DL\\DL-Lee\\HW1\\data\\train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

######## parse data
features = []
pms = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        features.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                features[471*i+j].append(data[t][480*i+j+s] )
        # the NO. 9th dimention is corresponding to PM2.5
        pms.append(data[9][480*i+j+9])
features = np.array(features)
pms = np.array(pms)

def feature_scaling(x):
     mean_value = x.mean(axis=0)
     std_value = x.std(axis=0)
     x = (x - mean_value) / std_value
     return x
features = feature_scaling(features)
pms = feature_scaling(pms)
 
#print(features[0:2,:])
batch_size = 471
n_batch = features.shape[0] // batch_size
#for batch in range(3):
#    batch_xs = features[batch:,:] # 162*1
#    batch_ys = pms[batch]
#    print(len(batch_xs))
#    print(batch_ys)


#
## define 2 placeholders for data(image) and label
x = tf.placeholder(tf.float32, [None, 162])
y = tf.placeholder(tf.float32, [None,1])  #(number 0 - 9)
## creat simpel network
W = tf.Variable(tf.zeros([162,1]))
b = tf.Variable(tf.zeros([1]))
prediction = tf.matmul(x,W)+b
## quadratic cost function
loss = tf.reduce_mean(tf.square(y-prediction))
## gradient descent 
train_step = tf.train.AdagradOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
## if equal- true, not - false
#correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(prediction,1))
### if prediction is correct, true, correct_prediction = 1, 
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


test_x = []
n_row = 0
text = open('F:\\ML-DL\\DL-Lee\\HW1\\data\\test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

saver = tf.train.Saver()
#
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            sess.run(train_step, feed_dict={
                    x: np.reshape(features[batch*batch_size:(batch+1)*batch_size,:],[batch_size,162]), 
                    y: np.reshape(pms[batch*batch_size:(batch+1)*batch_size],[batch_size,1])})
##    print('b:' + str(sess.run(b)) + ' || W:' + str(sess.run(W)))

    #
#    save_path = saver.save(sess, "F:\\ML-DL\\DL-Lee\\HW1\\result\\model.ckpt")
#    print("Model saved in path: %s" % save_path)
    pred = sess.run(prediction, feed_dict={x:test_x})


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    ans[i].append(pred[i].item())
#print(ans)

filename = "F:\\ML-DL\\DL-Lee\\HW1\\result\\predict-tf.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()









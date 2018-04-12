# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:57:17 2018

@author: Shiyao Han
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 
import tensorflow as tf
import csv
from sklearn.preprocessing import Imputer

# read and parse train data
data = pd.read_csv("F:\\ML-DL\\DL-Lee\\HW2\\data\\train.csv")

for i in range(data.shape[1]):
    col = data.iloc[:,i]
    try:
        series = col.str.find("?")
        idx = series[series==1]
        for j in range(len(idx)):
            a = idx.index[j]
#            print(data.iloc[idx.index[j],i])
            data.iloc[a,i] = data.iloc[0,i]
    except:
        print('!!')
            

x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1:data.shape[1]]
x_train = np.array(x.select_dtypes(include=[np.number]))
# encode the features
X = x.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
X_2 = X.apply(le.fit_transform)
X_2.head()
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
x_train = np.hstack((x_train, np.array(enc.transform(X_2).toarray())))
Y = y.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
y_train = Y.apply(le.fit_transform)




# read and parse test data
test = pd.read_csv("F:\\ML-DL\\DL-Lee\\HW2\\data\\test.csv")
for i in range(test.shape[1]):
    col = test.iloc[:,i]
    try:
        series = col.str.find("?")
        idx = series[series==1]
        for j in range(len(idx)):
            a = idx.index[j]
            test.iloc[a,i] = test.iloc[0,i]
    except:
        print('!!')
#    
x_test = np.array(test.select_dtypes(include=['number']))
# encode the features
X = test.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
X_2 = X.apply(le.fit_transform)
X_2.head()
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
bb = np.array(enc.transform(X_2).toarray())
x_test = np.hstack((x_test, np.array(enc.transform(X_2).toarray())))


X = pd.DataFrame(test['native_country'])
le = preprocessing.LabelEncoder()
X_2 = X.apply(le.fit_transform)
X_2.head()
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
b = np.array(enc.transform(X_2).toarray())
X = pd.DataFrame(x['native_country'])
le = preprocessing.LabelEncoder()
X_2 = X.apply(le.fit_transform)
X_2.head()
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
c = np.array(enc.transform(X_2).toarray())
print(b.shape[1])
print(c.shape[1])


print(y_train.shape)
# build the net
batch_size = 100
n_batch = x_train.shape[0] // batch_size
##
##
#### define 2 placeholders for data and label
x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
y = tf.placeholder(tf.float32, [None,y_train.shape[1]])  #(number 0 - 9)
## creat simpel network
W = tf.Variable(tf.zeros([x_train.shape[1],y_train.shape[1]]))
b = tf.Variable(tf.zeros([y_train.shape[1],1]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)
loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.AdagradOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
#
##
##
##
###
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1):
        for batch in range(n_batch):
            sess.run(train_step, feed_dict={
                    x: np.reshape(x_train[batch*batch_size:(batch+1)*batch_size,:],[batch_size,x_train.shape[1]]), 
                    y: np.reshape(y_train[batch*batch_size:(batch+1)*batch_size],[batch_size,y_train.shape[1]])})
#    print('b:' + str(sess.run(b)) + ' || W:' + str(sess.run(W)))
#
#    #
##    save_path = saver.save(sess, "F:\\ML-DL\\DL-Lee\\HW1\\result\\model.ckpt")
##    print("Model saved in path: %s" % save_path)
    pred = sess.run(prediction, feed_dict={x:x_test})


#ans = []
#for i in range(len(x_test)):
#    ans.append(["id_"+str(i)])
#    ans[i].append(pred[i].item())
#print(ans)

#filename = "F:\\ML-DL\\DL-Lee\\HW1\\result\\predict-tf.csv"
#text = open(filename, "w+")
#s = csv.writer(text,delimiter=',',lineterminator='\n')
#s.writerow(["id","value"])
#for i in range(len(ans)):
#    s.writerow(ans[i]) 
#text.close()



# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:53:09 2018

@author: Shiyao Han
"""

import pandas as pd
import numpy as np
import scipy as sp
import math

data = pd.read_csv("F:\\ML-DL\\DL-Lee\\HW1\\data\\pokemon-challenge\\pokemon.csv")

#x = data.loc[:data.shape[0], ['Total', 'HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
#y = np.array(data.loc[:data.shape[0], 'Attack'])

x = data.loc[:560, ['HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
y = np.array(data.loc[:560, 'Attack'])


# new: add a bias
x['bias'] = np.random.rand(x.shape[0],1)



def adagrad(x, y, w, l_rate = 7, epoch = 10):
    round = 0
    x_t = x.transpose()
    s_gra = np.zeros(x.shape[1])###########
    while round < epoch:
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        gra = np.dot(x_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra/ada
        print ('iteration: %d | Cost: %f  ' % ( round,cost_a))
        round += 1 
    return w

def feature_scaling(x):
     mean_value = x.mean(axis=0)
     print(mean_value.shape)
     std_value = x.std(axis=0)
     print(std_value.shape)
     x = x.subtract(mean_value, axis='columns').divide(std_value, 'columns')
#     print(x)
     return x
w = np.random.rand(x.shape[1])
#w = np.zeros(x.shape[1])
print(w)

#hypo = np.dot(x,w)
#x = feature_scaling(x)
print(x.shape)
w = adagrad(x, y, w, 10, 1000)

#print(scaled_feature)
    
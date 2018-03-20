# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 17:48:28 2018

@author: Shiyao Han
"""

#import csv
txt_file = "F:\\ML-DL\\DL-Lee\\HW0\\data\\words.txt"
content=open(txt_file).read()
ls=content.split(" ")
orderindex=0
for i in range(len(ls)):
    if ls.index(ls[i])==i:
        print("%s %s %s"%(ls[i],orderindex,ls.count(ls[i])))
        orderindex=orderindex+1
        
        
        
        
        
#list.index -> return all the indexes whose obj(?) is the same as given obj.
#list.count -> return the number of obj that have the same value of given obj
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:21:14 2018

@author: Shiyao Han
"""
#from PIL import Image
import cv2 

img_path = "F:\\ML-DL\\DL-Lee\\HW0\\data\\westbrook.jpg"
merge_path = "F:\\ML-DL\\DL-Lee\\HW0\\data\\new.jpg"

img = cv2.imread(img_path)
b, g, r = cv2.split(img)
b = b/2
r = r/2
g = g/2
merged = cv2.merge([b,g,r])


cv2.imwrite(merge_path,merged)



#img = Image.open(img_path)



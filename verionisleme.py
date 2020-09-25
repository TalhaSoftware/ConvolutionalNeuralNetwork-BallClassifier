# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:30:01 2020

@author: Talha Yazilim
"""

import cv2
import numpy as np

def resimal(dosyaadi):
    resim = cv2.imread("{}".format(dosyaadi))
    return resim

girisverisi = np.array([])
for i in range(30):
    i = i+1
    string = "dataset/{}.jpg".format(i)
    gelenresim = resimal(string)
    boyutluresim = cv2.resize(gelenresim,(224,224))
    girisverisi = np.append(girisverisi,boyutluresim)

girisverisi = np.reshape(girisverisi,(30,224,224,3))

print(girisverisi)
print(girisverisi.shape)

np.save("girisverimiz",girisverisi)













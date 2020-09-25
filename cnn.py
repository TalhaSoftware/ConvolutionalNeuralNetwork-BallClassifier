# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:36:47 2020

@author: Talha Yazilim
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import  Flatten,Dense,Dropout,Conv2D,MaxPooling2D,Activation
from keras.optimizers import RMSprop





girisverisi = np.load("girisverimiz.npy")

cikisverisi = np.array([  [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0], [0,1],[0,1],[0,1],[0,1],[0,1], [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1] ])             

splitverisi = girisverisi[1:3]
splitverisi = np.append(splitverisi,girisverisi[24:26])
splitcikis = np.array([ [1,0],[1,0],[1,0], [0,1], [0,1],[0,1]])
splitcikis = np.resize(splitcikis,(3,2))
splitverisi = np.resize(splitverisi,(-1,224,224,3))

"""
model = Sequential()
model = model.add(Conv2D(50,5,strides=(2,2),input_shape=(224,224,3)))
model = model.add(MaxPooling2D(5,5))
model = model.add(Conv2D(50,4))
model = model.add(Conv2D(50,3))
model = model.add(Conv2D(50,2))
model = model.add(Conv2D(50,1))
model = model.add(Flatten())

model = model.add(Dense(4096,activation="relu"))
model = model.add(Dropout(0.2))

model = model.add(Dense(4096,activation="relu"))
model = model.add(Dense(2))
model = model.add(Activation("softmax"))
"""

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))




model.compile(loss="binary_crossentropy",optimizer=RMSprop(lr=0.000001),metrics=["accuracy"])
model.summary()

model.fit(girisverisi,cikisverisi,batch_size=2,epochs=1,validation_data=(splitverisi,splitcikis))






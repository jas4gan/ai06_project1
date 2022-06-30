# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:05:47 2022

@author: jases

Github Project 1 

1. You are going to predict whether a patient has heart disease or not

2. LINK to the dataset 
    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

3. Criteria :
    a. The model should reach at least 90% accuracy for both training and validation.
    b. The model should not overfit 
    (validation loss needs to be within 10% difference with the training loss)
    
4. You required to upload onto your Github

"""
#1. Import packages
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


#2. Read the data from the csv file
file_path = r"C:\Users\jases\Desktop\AI-06\DL\data\heart.csv"
data = pd.read_csv(file_path)

#%%

x= data.drop('target',axis=1)
y= data['target']


SEED = 12345
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=SEED)


#Perform data normalization
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

#Build a feedforward NN with 3 hidden layers
model = keras.Sequential()
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1))

es = EarlyStopping(monitor='val_loss',patience=5)

#Compile the model
model.compile(optimizer='adam',loss='mae',metrics=['accuracy'])

#Train the model
batch_size=16
epochs=50
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=epochs,callbacks=[es])
#%%

#Visualize the loss and accuracy
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = history.epoch

plt.plot(epochs,training_loss,label='Training Loss')
plt.plot(epochs,val_loss,label='Validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs,training_acc,label='Training Accuracy')
plt.plot(epochs,val_acc,label='Validation Accuracy')
plt.legend()
plt.figure()

plt.show()

#%%
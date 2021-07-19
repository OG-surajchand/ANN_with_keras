# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:19:53 2021

@author: sctha
"""

#DataPreprocessing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

df = pd.read_csv('Churn_Modelling.csv')
x = df.iloc[:,3:13].values
y = df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,train_size=0.75,random_state=88)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#ANN MODEL BUILDING

ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(x_test)
y_pred = y_pred>0.5

new_data = np.array([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
new_data = sc.transform(new_data)
new_prediction = ann.predict(new_data)
new_prediction = new_prediction>0.5

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred)

    
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=6,activation='relu'))
    model.add(tf.keras.layers.Dense(units=6,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=build_model,batch_size=10,epochs=100)

accuracy = cross_val_score(estimator=model,X = x_train,y=y_train,cv=10)
mean = accuracy.mean()
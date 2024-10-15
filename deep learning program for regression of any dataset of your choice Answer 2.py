# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:33:30 2024

@author: DELL
"""

####### Write a deep learning program for regression of any dataset of your choice  ####

### deep learning regression model using the Boston Housing dataset

### step 1 : Import Libraries

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


### Step 2: Load and Prepare the Dataset

# Load the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Split the data into features and target
X = data.drop(columns=['medv'])
y = data['medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



## Step 3: Build the Model


# Define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#model.compile(optimizer='adam', loss='mse', metrics=['mae'])


###Step 4: Train the Model

# Train the model
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=1)


## Step 5: Evaluate the Model

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')



## step 6 : Make Predictions


# Make predictions


y_pred = model.predict(X_test)
print(y_pred[:20])  # Print the first 20 predictions


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()














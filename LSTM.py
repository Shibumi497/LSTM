#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:48:21 2021

@author: shibumi497
"""
##############################################
### LSTM for Energy consumption Prediction ###
##############################################


# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

print(np.version)
print(tf.__version__)

import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------
# Visualization
import matplotlib.pyplot as plt
import pandas as pd
# Load data
data = pd.read_csv('...')
data =data.dropna()
# Plot
plt.figure(figsize=(14.8, 4.2))
x = range(len(data['CCH_BC']))
plt.plot(x, data['CCH_BC'])
plt.xticks(x, data['Datetime'])
plt.xlabel('Datetime')
plt.ylabel('CCH_BC')
plt.show()
#-----------------------------------------------------------------------------

## First only with CCH BC ##

# Importing dataset

dataset_train = pd.read_csv('...')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 4392 timesteps and 1 output

X_train = []
y_train = []
for i in range(2196, 8760):
    X_train.append(training_set_scaled[i-2196:i, 0])  #taking into account the last 3 months
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping (add dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

## Building the LSTM

# Importing the Keras libraries and packages
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.007), loss = 'mean_absolute_error')

#Implementing early stop
import keras
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20 )

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, callbacks=[es_callback] )
import time
from datetime import timedelta
start_time = time.monotonic()

history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, callbacks=[es_callback] )

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

# Save the weights model
regressor.save_weights('C:/Users/.../lstm22_weights_cch_bc.h5')
#Out[19]: <keras.callbacks.callbacks.History at 0x27ba3bacf98>
regressor.load_weights('C:/Users/.../lstm20_weights_cch_bc.h5')
## Making the predictions and visualising the results

dataset_test = pd.read_csv('C:/Users/.../CCH_BC_2019_cch_bc.csv')
test_set = dataset_test.iloc[1:25, 1:2].values

# Getting the predicted CCH BC 2019
dataset_total = pd.concat((dataset_train['CCH_BC'], dataset_test['CCH_BC']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 2196:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(2196, 2220):
    X_test.append(inputs[i-2196:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_cch_bc = regressor.predict(X_test)
predicted_cch_bc = sc.inverse_transform(predicted_cch_bc)

# Visualising the results
plt.figure(figsize=(14.8, 4.2))
plt.plot(test_set, color = 'red', label = 'Real CCH BC 2019')
plt.plot(predicted_cch_bc, color = 'blue', label = 'Predicted CCH BC 2019')
plt.title('CCH BC Prediction a day ahead')
plt.xlabel('Time [in hours]')
plt.ylabel('CCH BC [in MWh]')
plt.legend()
plt.show()

# use a gray background
ax = plt.axes(axisbg='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
    
# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
plt.legend()
plt.show()

for i in range(len(predicted_cch_bc)):
	print("X=%s, Predicted=%s" % (predicted_cch_bc[i], test_set[i]))

print(predicted_cch_bc)
print(test_set)


## Metrics

# RMSE
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test_set, predicted_cch_bc, squared= False)
print(f"The RMSE is     {round(rmse,2)}")
      # MSE %
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(test_set, predicted_cch_bc)
print(f"The Mean Absolute Percentage Error is     {round(mape,2)}")

from sklearn.metrics import r2_score
r2 = r2_score(test_set, predicted_cch_bc)
print(f"The R2 is     {round(r2,2)}")

































#Visualise training and validation loss

plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#To get duration of the computation
import time
from datetime import timedelta
start_time = time.monotonic()


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))






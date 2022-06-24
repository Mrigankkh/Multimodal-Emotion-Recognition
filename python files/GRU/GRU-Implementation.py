#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization

def npz_to_dataFrame():
    label = 0
    df = pd.DataFrame(columns = ["Name", "Label"])
    for file_name in file:
        #if file_name[10] == '2':
        if file_name[7] == '1' or file_name[7] == '2' or file_name[7] == '3':
            label = 0
        elif file_name[7] == '5' or file_name[7] == '7':
            label = 1
        numpy_data = np.array([[file_name,label]])
        df2 = pd.DataFrame(data = numpy_data, columns = ["Name", "Label"])
        df = df.append(df2, ignore_index = True)   
    return(df)
    
    
file = os.listdir("/Users/sanjay/Desktop/Audio-Embedding/npz-values/")    
df = npz_to_dataFrame()
df1 = df.drop([210,1088])
df = df1
print(len(df))
y = df.Label
x = df.drop('Label', axis = 1)
print(y)


# In[2]:


def get_data(df, max_len):
    numpy_array = []
    numpy_array_2 = []
    zero_array = np.array([0]*512)
    for i in range(len(df)):
        try:
            numpy_array_name = df.iloc[i].Name
            numpy_array = np.load("/Users/sanjay/Desktop/Audio-Embedding/npz-values/" + numpy_array_name, allow_pickle=True)
            numpy_array_1 = numpy_array.f.arr_0
            numpy_array_2.append(numpy_array_1)
            
        except:
            print(i)
    
    
    for i in range(len(numpy_array_2)):
        numpy_array_2[i] = np.transpose(numpy_array_2[i])
        if max_len - len(numpy_array_2[i]) != 0:
            for j in range((max_len - len(numpy_array_2[i]))):
                numpy_array_2[i] = np.append(numpy_array_2[i],[zero_array],axis = 0)
        numpy_array_2[i] = np.transpose(numpy_array_2[i])
    return numpy_array_2
    


# In[3]:


def get_maxlen(df):
    numpy_array = []
    numpy_array_2 = []
    for i in range(len(df)):
            numpy_array_name = df.iloc[i].Name
            numpy_array = np.load("/Users/sanjay/Desktop/Audio-Embedding/npz-values/" + numpy_array_name, allow_pickle=True)
            numpy_array_1 = numpy_array.f.arr_0
            numpy_array_2.append(numpy_array_1)
    max_len = 0
    for i in range(len(numpy_array_2)):
        for j in range(len(numpy_array_2[i])):
            if len(numpy_array_2[i][j]) > max_len:
                max_len = len(numpy_array_2[i][j])
    return max_len, numpy_array_2


# In[4]:


max_len , numpy_array_2 = get_maxlen(df)
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2)

x_train = get_data(x_tr, max_len)
x_test = get_data(x_ts, max_len)





# In[5]:


print(max_len)

x_train = np.array(x_train).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)
y_train = np.array(y_tr).astype(np.float32)
y_test = np.array(y_ts).astype(np.float32)
print(np.shape(x_train))
#print(y_train.head())


# In[ ]:





# In[6]:


'''from keras.utils import normalize, to_categorical

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

'''


# In[7]:





model = Sequential()

model.add(GRU(units = 128))
model.add(Dropout(0.5))


'''model.add(Dense((128), activation = 'relu'))
model.add(Dropout(0))


model.add(Dense((16), activation = 'relu'))
model.add(Dropout(0))'''


model.add(Dense((1), activation = 'softmax'))




opt = tf.keras.optimizers.Adam(lr=0.1, decay=1e-5)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

history = model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))
model.summary()


# In[10]:


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[11]:


print(type(y_train[0]))


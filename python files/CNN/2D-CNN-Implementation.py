#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
from random import shuffle
import os
import librosa
import torch
import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from fairseq.models.wav2vec import Wav2VecModel
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[2]:


input_length = 16000*5

batch_size = 32




# In[3]:


def pre_processing(audio, sample_rate=16000):
    #file = "/Users/sanjay/Desktop/Audio-Embedding/ravdess-emotional-speech-audio/Actor_"+str(count)+"/" + file_name
    cp = torch.load('/Users/sanjay/Desktop/Audio-Embedding/wav2vec_large.pt' , map_location = torch.device('cpu'))
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.eval()
    signal , sr = librosa.load(audio, sr = 48000)
    signal_16khz = librosa.resample(signal, sr, 16000)
    wav_input_16khz = torch.from_numpy(signal_16khz).unsqueeze(0)
    z = model.feature_extractor(torch.Tensor(wav_input_16khz))
    c = model.feature_aggregator(z)
    c = c.cpu().detach().numpy()
    ct = c[0]
    return ct


# In[4]:


def load_audio_file(file_path, input_length=input_length):
    #data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
    cp = torch.load('/Users/sanjay/Desktop/Audio-Embedding/wav2vec_large.pt' , map_location = torch.device('cpu'))
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.eval()
    signal , sr = librosa.load(file_path, sr = 48000)
    signal_16khz = librosa.resample(signal, sr, 16000)
    wav_input_16khz = torch.from_numpy(signal_16khz).unsqueeze(0)
    z = model.feature_extractor(torch.Tensor(wav_input_16khz))
    c = model.feature_aggregator(z)
    c = c.cpu().detach().numpy()
    ct = c[0]
    data = ct
    if len(data)>input_length:
        
        
        max_offset = len(data)-input_length
        
        offset = np.random.randint(max_offset)
        
        data = data[offset:(input_length+offset)]
        
        
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0
        
        
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        
        
    
    return data
    


# In[5]:


train_files = glob.glob("/Users/sanjay/Desktop/Audio-Embedding/input/audio_train/*.wav")
test_files = glob.glob("/Users/sanjay/Desktop/Audio-Embedding/input/audio_test/*.wav")
train_labels = pd.read_csv("/Users/sanjay/Desktop/Audio-Embedding/input/train.csv")


# In[6]:


file_to_label = {"../input/audio_train/"+k:v for k,v in zip(train_labels.Name.values, train_labels.Label.values)}


# In[5]:


#file_to_label


# In[7]:

#
# data_base = load_audio_file(train_files[0])
# fig = plt.figure(figsize=(14, 8))
# plt.title('Raw wave : %s ' % (file_to_label[train_files[0]]))
# plt.ylabel('Amplitude')
# plt.plot(np.linspace(0, 1, input_length), data_base)
# plt.show()


# In[8]:


list_labels = sorted(list(set(train_labels.Label.values)))


# In[9]:


label_to_int = {k:v for v,k in enumerate(list_labels)}


# In[10]:


int_to_label = {v:k for k,v in label_to_int.items()}


# In[11]:


file_to_int = {k:label_to_int[v] for k,v in file_to_label.items()}


# In[7]:


def get_model_mel():

    nclass = len(list_labels)
    inp = Input(batch_shape=(None,None,512,batch_size))
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(16, kernel_size=(3, 7), activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 7))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[8]:


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[9]:



def train_generator(list_files, batch_size=batch_size):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, :,np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)
            
            yield batch_data, batch_labels


# In[10]:


tr_files, val_files = train_test_split(sorted(train_files), test_size=0.1, random_state=42)


# In[ ]:





# In[11]:


model = get_model_mel()


# In[ ]:


model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





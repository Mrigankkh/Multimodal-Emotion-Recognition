#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import librosa
import pandas as pd
import torch
import tensorflow as tf
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel
from numpy import savez_compressed

def npz_converter(file_name, count):
    file = "/Users/sanjay/Desktop/Audio-Embedding/ravdess-emotional-speech-audio/Actor_"+str(count)+"/" + file_name
    signal , sr = librosa.load(file, sr = 48000)
    signal_16khz = librosa.resample(signal, sr, 16000)
    wav_input_16khz = torch.from_numpy(signal_16khz).unsqueeze(0)
    z = model.feature_extractor(torch.Tensor(wav_input_16khz))
    c = model.feature_aggregator(z)
    c = c.cpu().detach().numpy()
    ct = c[0]
    file_name = file_name[:-4]
    savez_compressed(file_name+".npz", ct)
    
    
    
count = 10
while count <= 24:
    cp = torch.load('/Users/sanjay/Desktop/Audio-Embedding/wav2vec_large.pt' , map_location = torch.device('cpu'))
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.eval()

    array = os.listdir("/Users/sanjay/Desktop/Audio-Embedding/ravdess-emotional-speech-audio/Actor_"+str(count) )

    for file_name in array:
        npz_converter(file_name, count)
    count += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





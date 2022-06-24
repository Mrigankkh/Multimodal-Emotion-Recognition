#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch
import tensorflow as tf
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel

import librosa 



cp = torch.load('/Users/sanjay/Desktop/Audio-Embedding/wav2vec_large.pt' , map_location = torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()
file = "music-1.wav"
signal , sr = librosa.load(file, sr = 16000)
wav_input_16khz = torch.from_numpy(signal).unsqueeze(0)
z = model.feature_extractor(torch.Tensor(wav_input_16khz))
c = model.feature_aggregator(z)
print(c.size())


# In[66]:


c.size()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





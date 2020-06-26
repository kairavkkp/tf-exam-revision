#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[18]


import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numba import cuda
import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# In[4]:


headlines=[]
is_sarcastic = []

file = open("../data/sarcasm_detection/Sarcasm_Headlines_Dataset_v2.json")

for line in file.readlines():
    is_sarcastic.append(json.loads(line).get('is_sarcastic'))
    headlines.append(json.loads(line).get('headline').strip())
    
    
print(len(headlines))
print(len(is_sarcastic))
    


# In[6]:





# In[7]:


X_train,X_test,y_train,y_test = tts(headlines,is_sarcastic,test_size = 0.2,random_state = 101)


# In[8]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[9]:


BATCH_SIZE = 16

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index


# In[10]:


vocab_size = len(word_index)
vocab_size


# In[11]:


pad_type = 'post'
trunc_type = 'post'
max_len = max([len(x) for x in X_train])
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences,padding=pad_type,truncating=trunc_type,maxlen=max_len)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences,padding=pad_type,truncating=trunc_type,maxlen=max_len)


# In[12]:


max_len


# In[13]:


vocab_size


# In[14]:


embed_dim = 16
vocab_size = vocab_size
input_len = max_len


# In[15]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size+1,embed_dim,input_length=input_len))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(max_len,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()


# In[16]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[17]:


callback = ModelCheckpoint("sarcasm.h5",monitor='accuracy',mode='max')


# In[ ]:


history = model.fit(train_padded,y_train,epochs=10,validation_data=(test_padded,y_test),verbose=1,callbacks=[callback])


# In[ ]:



val_loss = history.history['val_loss']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
accuracy = history.history['accuracy']

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title("Validation vs Training Loss")
plt.legend(["Training Loss","Validation Loss"])
plt.savefig("Sarcasm Training vs Val Loss.png")
plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.title("Validation vs Training Acc")
plt.legend(["Training Accuracy","Validation Accuracy"])
plt.savefig("Sarcasm Training vs Val Acc.png")


# In[1]:


cuda.close()


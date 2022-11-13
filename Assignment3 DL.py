#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


x_train = np.loadtxt('input.csv', delimiter= ',')
y_train = np.loadtxt('labels.csv', delimiter= ',')

x_test = np.loadtxt('input_test.csv', delimiter= ',')
y_test = np.loadtxt('labels_test.csv', delimiter= ',')


# In[3]:


x_train =x_train.reshape(len(x_train), 100, 100, 3)
y_train =y_train.reshape(len(y_train), 1)
x_test =x_test.reshape(len(x_test), 100, 100, 3)
y_test =y_test.reshape(len(y_test), 1)

x_train = x_train / 255
x_test = x_test /255


# In[4]:


print("shape of x_train:", x_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of x_test:", x_test.shape)
print("shape of y_test:", y_test.shape)


# In[5]:


x_train[1, :]


# In[6]:


idx= random.randint(0, len(x_train))
plt.imshow(x_train[idx, :])
plt.show()


# In[7]:


model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation ='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation= 'sigmoid')
])


# In[8]:


model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation ='relu'))
model.add(Dense(1, activation= 'sigmoid'))


# In[9]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


model.fit(x_train, y_train, epochs=5, batch_size=64)


# In[11]:


model.evaluate(x_test, y_test)


# In[12]:


idx2 = random.randint(0, len(y_test))
plt.imshow(x_test[idx2, :])
plt.show()

y_pred = model.predict(x_test[idx2, :].reshape(1,100,100,3))
y_pred = y_pred > 0.5

if(y_pred == 0):
    pred = 'dog'
else:
    pred = 'cat'
print("our model says it is a :", pred) 


# In[13]:


history= model.fit(x_train, y_train, validation_data = (x_test, y_test),epochs=10)


# In[14]:


history.history.keys()


# In[15]:


accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']


# In[17]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(accuracy,label='training accuracy')
plt.plot(val_accuracy, label='val_accuracy')
plt.legend()
plt.title('training vs validation accuracy')

plt.subplot(2,2,2)
plt.plot(loss,label='training loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.title('training vs validation loss')
plt.show()


# In[ ]:





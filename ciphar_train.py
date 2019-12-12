#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.pyplot import imshow
import nn


# In[2]:


IMAGE_DIMENSION = 32
NO_OF_IMAGES = 10000
IP_IMG_PATH = 'data_batch_1B/B'
OP_IMG_PATH = 'data_batch_1C/C'

input_image = np.zeros((32*32, 10000))
output_image = np.zeros((32*32*3, 10000))
imageCount = 1000


# In[3]:


for i in range(10000,10000+imageCount):
    input_img = Image.open(IP_IMG_PATH + str(i) + '.png')
    output_img = Image.open(OP_IMG_PATH + str(i) + '.png')
    input_image[:,i-10000] = np.asarray(input_img, dtype=np.float128).flatten()
    output_image[:,i-10000] = np.asarray(output_img, dtype=np.float128).flatten()


# In[4]:


print(input_image.shape)
print(output_image.shape)


# In[5]:


input_image = input_image/float(255)
output_image = output_image/float(255)


# In[6]:


fig = plt.figure(figsize=(15,5))

for i in range(20):
    ax = fig.add_subplot(4, 5, 1 + i, xticks=[], yticks=[])
    im = input_image[:,i].reshape(IMAGE_DIMENSION,IMAGE_DIMENSION)
    plt.imshow(im)
plt.show()


# In[7]:


fig = plt.figure(figsize=(15,5))

for i in range(20):
    ax = fig.add_subplot(4, 5, 1 + i, xticks=[], yticks=[])
    im = output_image[:,i].reshape(IMAGE_DIMENSION,IMAGE_DIMENSION,3)
    plt.imshow(im)
plt.show()


# In[ ]:





# In[11]:


Nn = nn.NN(IMAGE_DIMENSION)


# In[12]:


Nn.initialize_parameters()
J = []


# In[ ]:

iterations = 2
for step in range(iterations):
    Nn.forward_prop(input_image)
    l2 = Nn.Loss(output_image)
    Nn.backward_prop(input_image, output_image)
    Nn.learning_algorithm(0.1)
    J.append(l2)
    print(step, l2)


# In[ ]:





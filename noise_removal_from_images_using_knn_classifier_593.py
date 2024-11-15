#!/usr/bin/env python
# coding: utf-8

# <font size=5>**Predicting Noisy Images**</font>

# <font size=3>**Import Libraries**</font>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import gzip
import matplotlib
import matplotlib.pyplot as plt


# <font size=3>**Define the Function to Show Images**</font>

# In[2]:


def showImage(data):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28)
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


# <font size=3>**Load the Data**</font>

# In[3]:


filePath_train_set = '/cxldata/datasets/project/mnist/train-images-idx3-ubyte.gz'
filePath_train_label = '/cxldata/datasets/project/mnist/train-labels-idx1-ubyte.gz'
filePath_test_set = '/cxldata/datasets/project/mnist/t10k-images-idx3-ubyte.gz'
filePath_test_label = '/cxldata/datasets/project/mnist/t10k-labels-idx1-ubyte.gz'


# In[4]:


with gzip.open(filePath_train_label, 'rb') as trainLbpath:
     trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                               offset=8)
with gzip.open(filePath_train_set, 'rb') as trainSetpath:
     trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
     testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8,
                               offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
     testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(testLabel), 784)


# In[5]:


X_train, X_test, y_train, y_test = trainSet, testSet, trainLabel, testLabel


# <font size=3>**Explore the Data**</font>

# In[6]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[7]:


showImage(X_train[0])
print("Output:", y_train[0])


# In[8]:


plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    array_image = X_train[i].reshape(28, 28)
    plt.imshow(array_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()


# <font size=3>**Shuffle the Data**</font>

# In[9]:


np.random.seed(42)


# In[10]:


shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# <font size=3>**Add Noise to the Data**</font>

# In[11]:


import numpy.random as rnd


# In[12]:


noise_train = rnd.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test


# <font size=3>**View the Noisy Data**</font>

# In[13]:


showImage(X_test_mod[4000])
showImage(y_test_mod[4000])


# In[14]:


showImage(X_train_mod[4000])
showImage(y_train_mod[4000])


# <font size=3>**Train a KNN Classifier**</font>

# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)


# <font size=3>**Predict Noisy Image**</font>

# In[17]:


clean_digit = knn_clf.predict([X_test_mod[5000]])


# In[18]:


showImage(clean_digit)


# In[19]:


showImage(y_test_mod[5000])


# <font size=5>**Author:**</font>

# -    **Prince Raj**

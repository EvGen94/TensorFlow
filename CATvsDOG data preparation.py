#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle 


# In[ ]:


DATADIR = "C:/Users/Evgenii/Downloads/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["dog","cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break        
        
   


# In[ ]:


IMG_SIZE = 100
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap="gray")
plt.show()


# In[ ]:


training_data = []

def create_training_data():
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        Class_num = CATEGORIES.index(category)  
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, Class_num]) 
            except Exception as e:
                pass
        
create_training_data()  


# In[ ]:


print(len(training_data))


# In[ ]:


random.shuffle(training_data)


# In[ ]:


for sample in training_data[0:10]:
    print(sample[1])


# In[ ]:


X = []
y = []


# In[ ]:


for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)    


# In[ ]:


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[ ]:





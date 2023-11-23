#!/usr/bin/env python
# coding: utf-8

# # DAT300 - Compulsory assignment 2

# ## Group 2
# Fight Club Goofy Edition
# 
# ## Orion username
# Insert your Orion username here
#   
# ## Members
# - Joel Yacob
# - Artush Mkrtchyan

# # Introduction

# Description of the compulsory assignment task as understood by the group (what kind of problem are you going to solve?, etc), and description of roles in the group during the compulsory assignment (80-120 words).

# # Data handling and visualisation

# In[1]:


# Import and extraction of data.
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.applications.vgg16 import VGG16 as vg, preprocess_input 


# Comments

# In[4]:


train_data = h5py.File("/mnt/courses/DAT300-22/CA2_data/train.h5")
test_data = h5py.File("/mnt/courses/DAT300-22/CA2_data/test.h5")


# In[5]:


print(train_data.keys())
print(test_data.keys())


# In[6]:


X = train_data["X"][:]
y = train_data["y"][:]

X_test_final = test_data["X"][:]


# In[7]:


print("Shape of X: ", X.shape)
print("Shape of y: ", y.shape)
print("Shape of X_test_final: ", X_test_final.shape)


# In[8]:


# Short exploration and visualisation of dataset (point 1 in Canvas).
print("Height of image: ",X.shape[1])
print("Width of image: ",X.shape[2])
print("Channels of image: ",X.shape[3])


# In[9]:


fig, ax = plt.subplots(1,5, figsize=(12,12))

for i , axis in enumerate(ax[:3]):
    axis.imshow(X[0][:,:,i])
    axis.title.set_text(f'Channel {i}')
    axis.set_xticks([])
    axis.set_yticks([])

ax[3].imshow(X[0])
ax[3].title.set_text("Full image")

ax[4].imshow(y[0])
ax[4].title.set_text("segmentation")

plt.show()


# Comments, describe what do you see from the visualisations

# # Methods

# Provide an overview of methods involved, strategies, number of parameters, failed efforts (160-240 words).

# # Preprocessing

# In[10]:


# Code for preprocessing of the data and transformation of labels for the binary problem (Point 2 in Canvas)

y = np.where(y != 0, 1,0) # Transform the train labels into a binary problem

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.1, random_state=69)


# # Results

# In[11]:


# Code and model training for your best Basic U-Net model (point 3 in Canvas)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


# def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2):
    
#     # Contracting Path
#     c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#     p1 = MaxPooling2D((2, 2))(c1)
#     p1 = Dropout(dropout)(p1)
    
#     c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
#     p2 = MaxPooling2D((2, 2))(c2)
#     p2 = Dropout(dropout)(p2)
    
#     c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(dropout)(p3)
    
#     c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(dropout)(p4)
    
#     c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
#     # Expansive Path
#     u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
#     u6 = concatenate([u6, c4])
#     u6 = Dropout(dropout)(u6)
#     c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
#     u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
#     u7 = concatenate([u7, c3])
#     u7 = Dropout(dropout)(u7)
#     c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
#     u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
#     u8 = concatenate([u8, c2])
#     u8 = Dropout(dropout)(u8)
#     c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
#     u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#     u9 = concatenate([u9, c1])
#     u9 = Dropout(dropout)(u9)
#     c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
#     outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
#     model = Model(inputs=[input_img], outputs=[outputs])
#     return model


# In[12]:


# input_img = Input(shape=(128,128,3))
# model = get_unet(input_img, n_filters = 32, dropout = 0.0, batchnorm = True, n_classes = 1)
# model.summary()


# In[13]:


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[14]:


# model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ['acc',f1_m]) # precision_m, recall_m
callback = EarlyStopping(monitor="loss", patience=5)


# In[15]:


# var = model.fit(X_train, y_train, epochs=35, callbacks = callback, validation_data=(X_test, y_test))


# Describe the results you are observing.

# In[16]:


# Code and model training for your best transfer learning model (point 4 in Canvas)

def get_unet_vg16(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2):
    
    encode_model = vg(input_tensor=input_img, include_top = False, weights="imagenet")
    
    for layer in encode_model.layers:
        layer.trainable = False
        
    encoder_output = encode_model.get_layer("block5_conv3").output
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(encoder_output)
    u6 = concatenate([u6, encode_model.get_layer("block4_conv3").output])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, encode_model.get_layer("block3_conv3").output])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, encode_model.get_layer("block2_conv2").output])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, encode_model.get_layer("block1_conv2").output])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs_vg = Conv2D(n_classes, (1, 1), padding = 'same', activation='sigmoid')(c9)
    vg_model = Model(inputs=[encode_model.input], outputs=[outputs_vg])
    return vg_model


# In[17]:


input_img = Input(shape=(128,128,3))
model_vg = get_unet_vg16(input_img, n_filters = 64, dropout = 0.2, batchnorm = True, n_classes = 1)
model_vg.summary()


# In[ ]:


inputs = Input((128,128,3))
model_vg.compile(optimizer = Adam(learning_rate = 0.001), loss = "binary_crossentropy", metrics = ['acc',f1_m]) # precision_m, recall_m
# Fit data to model
model_vg.fit(X_train, y_train,
         epochs=10,
         batch_size=42,
#         shuffle=True,
         validation_data=(X_test, y_test),
             callbacks= callback
         )


# Describe the results you are observing.

# In[ ]:


# Orion related code, i.e. slurm-script, username, code to access data on Orion and time usage for your modelling (point 5)


# Describe the results you are observing.

# In[ ]:


# Optional: Code and model training for multiclass segmentation with U-net (Point 5 in Canvas)


# Describe the results you are observing.
# 

# In[ ]:


predictions = model_vg.predict(X_test_final)
pred = predictions.flatten()
pred


# In[ ]:


submission_df = pd.DataFrame(data=list(range(len(pred))), 
                             columns=["Id"])

submission_df["Predicted"] = pred
submission_df = submission_df.round(0).astype("int")
                                    
submission_df['Predicted'] = np.where(submission_df['Predicted'] == 0,
                                      False, True)

submission_df.to_csv("CA2_goofy_submission.csv", index=False)
submission_df


# # Discussion / conclusion

# Provide a summary of the assignment: (you are required to address **the first three** points of the list below)
# - obstacles / problems you have met regarding the modelling proces
# - degree of success
# - given more time, what would be done differently
# - further comments (if any)
# 
# Please specify which of the models was used for your final Kaggle submission.

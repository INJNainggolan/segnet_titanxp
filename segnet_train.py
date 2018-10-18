#coding=utf-8

import h5py
#import win_unicode_console
#win_unicode_console.enable()
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras import metrics
from keras.models import *
from keras.layers import Conv2D,SeparableConv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Dropout,Layer
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras.layers.merge import concatenate
from keras import regularizers
from PIL import Image  
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 7  
np.random.seed(seed)

#data_shape = 360*480  
img_w = 256  
img_h = 256  
#有一个为背景  
n_label = 4+1  
  
classes = [0. ,  1.,  2.,   3.  , 4.]


EPOCHS = 30
BS = 15



learning_rate = 0.01
decay = 0.001
learning_rate = learning_rate * 1/(1 + decay * EPOCHS)
sgd = optimizers.SGD(lr = learning_rate, decay = learning_rate/EPOCHS, momentum=0.9, nesterov=True)



#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08)


labelencoder = LabelEncoder()  
labelencoder.fit(classes)  

#image_sets = ['1.png','2.png','3.png']

        
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


filepath ="/home/cqnu/dataset/Potsdam/Potsdam_500000/train/"
log_filepath = "/home/cqnu/output/segnet_Potsdam_output/logs"

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'src//' + url)
            img = img_to_array(img) 
            train_data.append(img)  
            label = load_img(filepath + 'label//' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'src//' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label//' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0


def SegNet_SeparableConv2D():
    inputs = Input((img_w, img_h, 3))  # maxl

    conv1_1 = SeparableConv2D(64, (3, 3), activation="relu", padding="same", name = 'conv1_1')(inputs)
    conv1_1_bn = BatchNormalization(name = 'conv1_1_bn')(conv1_1)
    conv1_2 = SeparableConv2D(64, (3, 3), activation="relu", padding="same", name = 'conv1_2')(conv1_1_bn)
    conv1_2_bn = BatchNormalization(name = 'conv1_2_bn')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'pool1')(conv1_2_bn)

    conv2_1 = SeparableConv2D(128, (3, 3), activation="relu", padding="same", name = 'conv2_1')(pool1)
    conv2_1_bn = BatchNormalization(name = 'conv2_1_bn')(conv2_1)
    conv2_2 = SeparableConv2D(128, (3, 3), activation="relu", padding="same", name = 'conv2_2')(conv2_1_bn)
    conv2_2_bn = BatchNormalization(name = 'conv2_2_bn')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'pool2')(conv2_2_bn)

    conv3_1 = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_1')(pool2)
    conv3_1_bn = BatchNormalization(name = 'conv3_1_bn')(conv3_1)
    conv3_2 = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_2')(conv3_1_bn)
    conv3_2_bn = BatchNormalization(name = 'conv3_2_bn')(conv3_2)
    conv3_3 = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_3')(conv3_2_bn)
    conv3_3_bn = BatchNormalization(name = 'conv3_3_bn')(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'pool3')(conv3_3_bn)
    encdrop3 = Dropout(0.5, name = 'encdrop3')(pool3)

    conv4_1 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_1')(encdrop3)
    conv4_1_bn = BatchNormalization(name = 'conv4_1_bn')(conv4_1)
    conv4_2 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_2')(conv4_1_bn)
    conv4_2_bn = BatchNormalization(name = 'conv4_2_bn')(conv4_2)
    conv4_3 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_3')(conv4_2_bn)
    conv4_3_bn = BatchNormalization(name = 'conv4_3_bn')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name = 'pool4')(conv4_3_bn)
    encdrop4 = Dropout(0.5, name = 'encdrop4')(pool4)

    conv5_1 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_1')(encdrop4)
    conv5_1_bn = BatchNormalization(name = 'conv5_1_bn')(conv5_1)
    conv5_2 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_2')(conv5_1_bn)
    conv5_2_bn = BatchNormalization(name = 'conv5_2_bn')(conv5_2)
    conv5_3 = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_3')(conv5_2_bn)
    conv5_3_bn = BatchNormalization(name = 'conv5_3_bn')(conv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2), name = 'pool5')(conv5_3_bn)
    encdrop5 = Dropout(0.5, name = 'encdrop5')(pool5)

    up5 = UpSampling2D(size=(2, 2), name = 'up5')(encdrop5)

    conv5_3_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_3_D')(up5)
    conv5_3_D_bn = BatchNormalization(name = 'conv5_3_D_bn')(conv5_3_D)
    conv5_2_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_2_D')(conv5_3_D_bn)
    conv5_2_D_bn = BatchNormalization(name = 'conv5_2_D_bn')(conv5_2_D)
    conv5_1_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv5_1_D')(conv5_2_D_bn)
    conv5_1_D_bn = BatchNormalization(name = 'conv5_1_D_bn')(conv5_1_D)
    decdrop5 = Dropout(0.5, name = 'decdrop5')(conv5_1_D_bn)

    up4 = concatenate([UpSampling2D(size=(2, 2))(decdrop5), conv4_3_bn], axis=-1, name = 'up4')  # maxl
    conv4_3_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_3_D')(up4)
    conv4_3_D_bn = BatchNormalization(name = 'conv4_3_D_bn')(conv4_3_D)
    conv4_2_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_2_D')(conv4_3_D_bn)
    conv4_2_D_bn = BatchNormalization(name = 'conv4_2_D_bn')(conv4_2_D)
    conv4_1_D = SeparableConv2D(512, (3, 3), activation="relu", padding="same", name = 'conv4_1_D')(conv4_2_D_bn)
    conv4_1_D_bn = BatchNormalization(name = 'conv4_1_D_bn')(conv4_1_D)
    decdrop4 = Dropout(0.5, name = 'decdrop4')(conv4_1_D_bn)

    up3 = concatenate([UpSampling2D(size=(2, 2))(decdrop4), conv3_3_bn], axis=-1, name = 'up3')  # maxl
    conv3_3_D = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_3_D')(up3)
    conv3_3_D_bn = BatchNormalization(name = 'conv3_3_D_bn')(conv3_3_D)
    conv3_2_D = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_2_D')(conv3_3_D_bn)
    conv3_2_D_bn = BatchNormalization(name = 'conv3_2_D_bn')(conv3_2_D)
    conv3_1_D = SeparableConv2D(256, (3, 3), activation="relu", padding="same", name = 'conv3_1_D')(conv3_2_D_bn)
    conv3_1_D_bn = BatchNormalization(name = 'conv3_1_D_bn')(conv3_1_D)
    decdrop3 = Dropout(0.5, name = 'decdrop3')(conv3_1_D_bn)

    up2 = concatenate([UpSampling2D(size=(2, 2))(decdrop3), conv2_2_bn], axis=-1, name = 'up2')  # maxl
    conv2_2_D = SeparableConv2D(128, (3, 3), activation="relu", padding="same", name = 'conv2_2_D')(up2)
    conv2_2_D_bn = BatchNormalization(name = 'conv2_2_D_bn')(conv2_2_D)
    conv2_1_D = SeparableConv2D(128, (3, 3), activation="relu", padding="same", name = 'conv2_1_D')(conv2_2_D_bn)
    conv2_1_D_bn = BatchNormalization(name = 'conv2_1_D_bn')(conv2_1_D)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2_1_D_bn), conv1_2_bn], axis=-1, name = 'up1')  # maxl
    conv1_2_D = SeparableConv2D(64, (3, 3), activation="relu", padding="same", name = 'conv1_2_D')(up1)
    conv1_2_D_bn = BatchNormalization(name = 'conv1_2_D_bn')(conv1_2_D)

    conv1_1_D = SeparableConv2D(n_label, (1, 1), activation="relu", padding="same", name = 'conv1_1_D')(conv1_2_D_bn)

    reshape = Reshape((img_w * img_h, n_label), name = 'reshape')(conv1_1_D)

    activation = Activation('softmax', name = 'activation')(reshape)

    model = Model(inputs=inputs, outputs=activation)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['mae', 'acc'])
    return model


def SegNet_pooling_indices():
    inputs = Input((img_w, img_h, 3))  # maxl

    conv1_1 = Conv2D(64, (3, 3), activation="relu", padding="same", name='conv1_1')(inputs)
    conv1_1_bn = BatchNormalization(name='conv1_1_bn')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation="relu", padding="same", name='conv1_2')(conv1_1_bn)
    conv1_2_bn = BatchNormalization(name='conv1_2_bn')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_2_bn)

    conv2_1 = Conv2D(128, (3, 3), activation="relu", padding="same", name='conv2_1')(pool1)
    conv2_1_bn = BatchNormalization(name='conv2_1_bn')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation="relu", padding="same", name='conv2_2')(conv2_1_bn)
    conv2_2_bn = BatchNormalization(name='conv2_2_bn')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_2_bn)

    conv3_1 = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_1')(pool2)
    conv3_1_bn = BatchNormalization(name='conv3_1_bn')(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_2')(conv3_1_bn)
    conv3_2_bn = BatchNormalization(name='conv3_2_bn')(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_3')(conv3_2_bn)
    conv3_3_bn = BatchNormalization(name='conv3_3_bn')(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_3_bn)
    encdrop3 = Dropout(0.5, name='encdrop3')(pool3)

    conv4_1 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_1')(encdrop3)
    conv4_1_bn = BatchNormalization(name='conv4_1_bn')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_2')(conv4_1_bn)
    conv4_2_bn = BatchNormalization(name='conv4_2_bn')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_3')(conv4_2_bn)
    conv4_3_bn = BatchNormalization(name='conv4_3_bn')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_3_bn)
    encdrop4 = Dropout(0.5, name='encdrop4')(pool4)

    conv5_1 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_1')(encdrop4)
    conv5_1_bn = BatchNormalization(name='conv5_1_bn')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_2')(conv5_1_bn)
    conv5_2_bn = BatchNormalization(name='conv5_2_bn')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_3')(conv5_2_bn)
    conv5_3_bn = BatchNormalization(name='conv5_3_bn')(conv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5_3_bn)
    encdrop5 = Dropout(0.5, name='encdrop5')(pool5)

    up5 = UpSampling2D(size=(2, 2), name='up5')(encdrop5)

    conv5_3_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_3_D')(up5)
    conv5_3_D_bn = BatchNormalization(name='conv5_3_D_bn')(conv5_3_D)
    conv5_2_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_2_D')(conv5_3_D_bn)
    conv5_2_D_bn = BatchNormalization(name='conv5_2_D_bn')(conv5_2_D)
    conv5_1_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv5_1_D')(conv5_2_D_bn)
    conv5_1_D_bn = BatchNormalization(name='conv5_1_D_bn')(conv5_1_D)
    decdrop5 = Dropout(0.5, name='decdrop5')(conv5_1_D_bn)

    up4 = concatenate([UpSampling2D(size=(2, 2))(decdrop5), conv4_3_bn], axis=-1, name='up4')  # maxl
    conv4_3_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_3_D')(up4)
    conv4_3_D_bn = BatchNormalization(name='conv4_3_D_bn')(conv4_3_D)
    conv4_2_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_2_D')(conv4_3_D_bn)
    conv4_2_D_bn = BatchNormalization(name='conv4_2_D_bn')(conv4_2_D)
    conv4_1_D = Conv2D(512, (3, 3), activation="relu", padding="same", name='conv4_1_D')(conv4_2_D_bn)
    conv4_1_D_bn = BatchNormalization(name='conv4_1_D_bn')(conv4_1_D)
    decdrop4 = Dropout(0.5, name='decdrop4')(conv4_1_D_bn)

    up3 = concatenate([UpSampling2D(size=(2, 2))(decdrop4), conv3_3_bn], axis=-1, name='up3')  # maxl
    conv3_3_D = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_3_D')(up3)
    conv3_3_D_bn = BatchNormalization(name='conv3_3_D_bn')(conv3_3_D)
    conv3_2_D = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_2_D')(conv3_3_D_bn)
    conv3_2_D_bn = BatchNormalization(name='conv3_2_D_bn')(conv3_2_D)
    conv3_1_D = Conv2D(256, (3, 3), activation="relu", padding="same", name='conv3_1_D')(conv3_2_D_bn)
    conv3_1_D_bn = BatchNormalization(name='conv3_1_D_bn')(conv3_1_D)
    decdrop3 = Dropout(0.5, name='decdrop3')(conv3_1_D_bn)

    up2 = concatenate([UpSampling2D(size=(2, 2))(decdrop3), conv2_2_bn], axis=-1, name='up2')  # maxl
    conv2_2_D = Conv2D(128, (3, 3), activation="relu", padding="same", name='conv2_2_D')(up2)
    conv2_2_D_bn = BatchNormalization(name='conv2_2_D_bn')(conv2_2_D)
    conv2_1_D = Conv2D(128, (3, 3), activation="relu", padding="same", name='conv2_1_D')(conv2_2_D_bn)
    conv2_1_D_bn = BatchNormalization(name='conv2_1_D_bn')(conv2_1_D)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2_1_D_bn), conv1_2_bn], axis=-1, name='up1')  # maxl
    conv1_2_D = Conv2D(64, (3, 3), activation="relu", padding="same", name='conv1_2_D')(up1)
    conv1_2_D_bn = BatchNormalization(name='conv1_2_D_bn')(conv1_2_D)

    conv1_1_D = Conv2D(n_label, (1, 1), activation="relu", padding="same", name='conv1_1_D')(conv1_2_D_bn)

    reshape = Reshape((img_w * img_h, n_label), name='reshape')(conv1_1_D)

    activation = Activation('softmax', name='activation')(reshape)

    model = Model(inputs=inputs, outputs=activation)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['mae', 'acc'])
    return model

def SegNet():  
    model = Sequential()  
    #encoder  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h, 3),padding='same',activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1),padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

     #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #(8,8)
    #decoder  
    model.add(UpSampling2D(size=(2,2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
    model.add(Reshape((img_w*img_h,n_label)))
    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)  
#    model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['mae', 'acc'])
#    tf.summary.scalar('learning_rate', learning_rate)
    model.summary()
    return model
'''
class RateHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.Rate = []

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
'''

  
def train(args):
#def train():

#    model = SegNet_pooling_indices()
#    model = SegNet()

    model = SegNet_SeparableConv2D()

    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',verbose=1, save_best_only=True,mode='max')
    tensorboard = TensorBoard(log_dir=log_filepath, histogram_freq=0, write_graph=True,
                                                      write_grads=True, write_images=True)
#    Reduce_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon = 0.0001, cooldown=0, min_lr=0)
#    history = LossHistory()
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)


    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                           verbose=1,
                           validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS, shuffle=True,
                           callbacks=[modelcheck, tensorboard], max_q_size=1)

#    print(history.losses)

    # plot the training loss and accuracy
    # save as JSON
    json_string = model.to_json()
    open('/home/cqnu/output/segnet_Potsdam_output/my_model_architecture.json','w').write(json_string)
    model = model_from_json(open('/home/cqnu/output/segnet_Potsdam_output/my_model_architecture.json').read())

    model.save_weights('/home/cqnu/output/segnet_Potsdam_output/my_model_weights.h5')




    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("/home/cqnu/output/segnet_Potsdam_output/segnet_loss_acc.png")
    

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", help="using data augment or not",
                    action="store_true", default=False)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args





if __name__=='__main__':


    args = args_parse()

    if args['augment'] == True:
        filepath ="/home/cqnu/dataset/Potsdam/Potsdam_500000/train/"
    train(args)



    #predict()  

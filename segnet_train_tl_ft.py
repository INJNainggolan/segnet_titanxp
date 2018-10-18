# coding=utf-8

import h5py
# import win_unicode_console
# win_unicode_console.enable()
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import *
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras import regularizers
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm



import tensorflow as tf
from keras import models
from keras.applications.inception_v3 import InceptionV3,preprocess_input


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)

# data_shape = 360*480
img_w = 256
img_h = 256
# 有一个为背景
n_label = 4 + 1

classes = [0., 1., 2., 3., 4.]

EPOCHS = 30
BS = 2



def SegNet():
    model = Sequential()
    # encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())

    #    model.add(Dropout(0.5))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (8,8)
    # decoder
    model.add(UpSampling2D(size=(2, 2)))

    # (16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))

    #    model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    # (32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    # (256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))
    # axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)
    #    model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

#base_model = sm.SegNet()#加载的应该是权重，而不是原来的model
base_model = load_model('/home/zq/3rd_segnet_epoch_30_bs_2_new_image_100000.h5')
'''
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='elu')(x)
predictions = Dense(17,activation='softmax')(x)
model = Model(input=base_model.input,output=predictions)
'''

model = SegNet()
#model = load_model('/home/zq/3rd_segnet_epoch_30_bs_2_new_image_100000.h5')

'''
learning_rate = 0.01
sgd = optimizers.SGD(lr = learning_rate, decay = learning_rate/EPOCHS, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08)
'''

labelencoder = LabelEncoder()
labelencoder.fit(classes)


# image_sets = ['1.png','2.png','3.png']


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


filepath = "/home/zq/dataset/RSI_train/segnet_train_50000/"


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
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
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
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
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)
                train_label = to_categorical(train_label, num_classes=n_label)
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

            # data for validation


def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
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
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label).flatten()
                valid_label = labelencoder.transform(valid_label)
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def setup_to_transfer_learning(model,base_model):


    for layer in base_model.layers[:15]:
        layer.trainable = False
    print("#####  transfer_learning  ########")
    print(tf.trainable_variables())
#    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#    model.summary()
    '''
    with tf.variable_scope('base_model'):
        base_model = load_model('/home/zq/3rd_segnet_epoch_30_bs_2_new_image_100000.h5')
#        base_model = InceptionV3(weights='imagenet', include_top=False)
    with tf.variable_scope('Conv2D'):
        x = base_model.output
        x = GlobalAveragePooling1D()(x)
        x = Dense(1024, activation='elu')(x)
        predictions = Dense(17, activation='softmax')(x)
        model = Model(input=base_model.input, output=predictions)
    trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv2D')
    print(trainable_var)
    '''




def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    print("#####  fine_tune_I  ########")
    print(tf.trainable_variables())
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    print("#####  fine_tune_II  ########")
    print(tf.trainable_variables())
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()


def train():

#    setup_to_transfer_learning(model,base_model)

#    model = load_model('/home/zq/3rd_segnet_epoch_30_bs_2_new_image_100000.h5')

   

    model.trainable = True
    for layer in model.layers[:15]:
        layer.trainable = False
    model.summary()
    '''



    model = model_from_json(open('my_model_architecture.json').read())

    model.trainable = True
    for layer in model.layers[:20]:
        layer.trainable = False

    model.load_weights('my_model_weights.h5')

    model.summary()
    '''
    '''


    model = load_model('/home/zq/3rd_segnet_epoch_30_bs_2_new_image_100000.h5')
    #        base_model = InceptionV3(weights='imagenet', include_top=False)
    with tf.variable_scope('Conv2D'):
        x = base_model.output
        x = GlobalAveragePooling1D()(x)
        x = Dense(1024, activation='elu')(x)
        predictions = Dense(17, activation='softmax')(x)
        model = Model(input=base_model.input, output=predictions)
    trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv2D')
    print(trainable_var)
    
    '''



    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    H_1 = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=callable, max_q_size=1)


    print("#####  fine_tune_I  ########")
    print(tf.trainable_variables())
    model.save("/home/zq/output/segnet_output/tl.h5")

    # plot the training loss and accuracy
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
    plt.savefig("/home/zq/output/segnet_output/segnet_loss_acc_tl.png")

    setup_to_fine_tune(model,base_model)

    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    H_2 = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=callable, max_q_size=1)

    model.save("/home/zq/output/segnet_output/ft.h5")

    # plot the training loss and accuracy
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
    plt.savefig("/home/zq/output/segnet_output/segnet_loss_acc_ft.png")




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


if __name__ == '__main__':

    args = args_parse()

    if args['augment'] == True:
        filepath = "/home/zq/dataset/RSI_train/segnet_train_50000/"


    train()

    # predict()

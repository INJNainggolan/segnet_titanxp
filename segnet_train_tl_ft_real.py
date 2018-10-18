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
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Dropout, \
    Layer
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
from keras.callbacks import TensorBoard

from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.optimizers import Adagrad

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 7
np.random.seed(seed)

# data_shape = 360*480
img_w = 256
img_h = 256
# 有一个为背景
n_label = 4 + 1

classes = [0., 1., 2., 3., 4.]

log_filepath_tl = '/home/zq/output/segnet_output_tl_ft/logs_tl/log'
log_filepath_ft = '/home/zq/output/segnet_output_tl_ft/logs_ft/log'

EPOCHS = 30
EPOCHS_tl = 2
EPOCHS_ft = 20
BS = 2

learning_rate = 0.1
sgd = optimizers.SGD(lr = learning_rate, decay = learning_rate/EPOCHS, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08)


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


filepath = "/home/zq/dataset/dataset_RSI_eCognition/train/"


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
    for layer in base_model.layers:
        layer.trainable = False
    tf.trainable_variables()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics= ['mae', 'acc'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 52
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics= ['mae', 'acc'])


def train(args):
    base_model = load_model('/home/zq/9th_15000.h5')
    model = base_model
    '''
    setup_to_transfer_learning(model, base_model)
    model.summary()
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    H_tl = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS_tl,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=[TensorBoard(log_dir=log_filepath_tl, histogram_freq=0, write_graph=True,
                                                          write_grads=True, write_images=True)], max_q_size=1)

    # save as JSON
    json_string = model.to_json()
    open('/home/zq/output/segnet_output_tl_ft/my_model_architecture_tl.json', 'w').write(json_string)
    model = model_from_json(open('/home/zq/output/segnet_output_tl_ft/my_model_architecture_tl.json').read())

    model.save_weights('/home/zq/output/segnet_output_tl_ft/my_model_weights_tl.h5')

    model.save('/home/zq/output/segnet_output_tl_ft/segnet_tl.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS_tl
    plt.plot(np.arange(0, N), H_tl.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H_tl.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H_tl.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H_tl.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch_tl #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("/home/zq/output/segnet_output_tl_ft/segnet_loss_acc_tl.png")
    '''

    setup_to_fine_tune(model, base_model)
    model.summary()
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=log_filepath_ft, histogram_freq=0, write_graph=True,
                              write_grads=True, write_images=True)

    H_ft = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS_ft,
                               verbose=1,
                               validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                               callbacks=[modelcheck, tensorboard], max_q_size=1)

    # save as JSON
    json_string = model.to_json()
    open('/home/zq/output/segnet_output_tl_ft/my_model_architecture_ft.json', 'w').write(json_string)
    model = model_from_json(open('/home/zq/output/segnet_output_tl_ft/my_model_architecture_ft.json').read())

    model.save_weights('/home/zq/output/segnet_output_tl_ft/my_model_weights_ft.h5')

    model.save('/home/zq/output/segnet_output_tl_ft/segnet_ft.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS_ft
    plt.plot(np.arange(0, N), H_ft.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H_ft.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H_ft.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H_ft.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch_ft #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("/home/zq/output/segnet_output_tl_ft/segnet_loss_acc_ft.png")


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
    train(args)

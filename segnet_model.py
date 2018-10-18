# coding=utf-8

import matplotlib

matplotlib.use("Agg")
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Dropout, \
    Layer

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
    return model
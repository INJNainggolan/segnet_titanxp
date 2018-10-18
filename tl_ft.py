from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.callbacks import TensorBoard
import tensorflow as tf


#数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory='/home/zq/dataset/flowers17/train',
                                                    target_size=(299,299),
                                                    batch_size=64)
val_generator = val_datagen.flow_from_directory(directory='/home/zq/dataset/flowers17/validation',
                                                    target_size=(299,299),
                                                    batch_size=64)


#构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False)

#添加新的输出层

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
predictions = Dense(17,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
#plot_model(model,'/home/zq/output/flower17/tlmodel.png')

#model = base_model

'''
这里的base_model和model里面的iv3都指向同一个地址
'''
def setup_to_transfer_learning(model,base_model):
    for layer in base_model.layers:
        layer.trainable = False
    tf.trainable_variables()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['mae', 'acc'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    tf.trainable_variables()
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy', metrics= ['mae', 'acc'])


log_filepath_tl = '/home/zq/output/flower17/logs_tl'
log_filepath_ft = '/home/zq/output/flower17/logs_ft'


setup_to_transfer_learning(model,base_model)
model.summary()
history_tl = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=800,
                                 epochs=2,
                                 validation_data=val_generator,
                                 validation_steps=12,
                                 class_weight='auto', 
                                 callbacks = [TensorBoard(log_dir=log_filepath_tl, histogram_freq=0, write_graph=True,
                                                          write_grads=True, write_images=True)])

model.save('/home/zq/output/flower17/flowers17_iv3_tl.h5')

setup_to_fine_tune(model,base_model)
model.summary()
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=800,
                                 epochs=2,
                                 validation_data=val_generator,
                                 validation_steps=1,
                                 class_weight='auto',
                                 callbacks = [TensorBoard(log_dir=log_filepath_ft, histogram_freq=0, write_graph=True,
                                                          write_grads=True, write_images=True)])

model.save('/home/zq/output/flower17/flowers17_iv3_ft.h5')



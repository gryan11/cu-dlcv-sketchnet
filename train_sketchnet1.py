import os
import string
import h5py

import time, pickle, pandas

import numpy as np

import keras
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers
from keras import backend as K
import glob
import shutil

import tensorflow as tf

batch_size = 25

data_root = 'data/tu-berlin/'
train_data_dir = data_root + 'train'
validation_data_dir = data_root + 'test'

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
#         target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
#         target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')

batch, y = train_generator.next()
img_height = batch.shape[1]
img_width = batch.shape[2]
channels = batch.shape[3]
num_classes = len(y[0])
print num_classes, img_height, img_width, channels

def build_sketchnet1():
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), 
                            padding='same',
                            input_shape=(img_width, img_height, channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.0625))
    
    model.add(Convolution2D(256, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.125))
              
    model.add(Convolution2D(512, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(512, (3, 3), 
                            padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initialize optimizer
    opt = rmsprop(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model, opt

sketchnet1_full, opt_full = build_sketchnet1()
sketchnet1_full.summary()


model_name = 'sketchnet1_full'

sketchnet1_full.load_weights('./models/'+model_name+\
                             '.h5')

# set up logging

tensorboard_callback =\
    TensorBoard(log_dir='./logs/'+model_name+'/', 
                histogram_freq=0, 
                write_graph=False, write_images=False)
checkpoint_callback =\
    ModelCheckpoint('./models/'+model_name+'.h5', 
                    monitor='val_acc', verbose=0, 
                    save_best_only=True, 
                    save_weights_only=True, 
                    mode='auto', period=1)

total_epochs = 100
nb_epoch = 500
K.set_value(opt_full.lr, 0.01)

hist = sketchnet1_full.fit_generator(
        train_generator,
        steps_per_epoch = 530,
        epochs = nb_epoch + total_epochs,
        validation_data = validation_generator,
        validation_steps = 270,
#         nb_val_samples = nb_validation_samples,
        verbose = 1,
        initial_epoch = total_epochs,
        callbacks=[tensorboard_callback, 
                   checkpoint_callback]
)

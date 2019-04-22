import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.optimizers import SGD, Nadam, Adam, RMSprop

import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras import backend as K
from skimage import io, color
from collections import namedtuple

from models import build_model

import imgaug as ia
from imgaug import augmenters as iaa

run = wandb.init(project='colorizer-applied-dl')
config = run.config
# config = namedtuple("CONFIG", ["num_epochs", "batch_size", "img_dir", "height", "width", "learning_rate"])

config.num_epochs = 200
config.batch_size = 16
config.img_dir = "images"
config.height = 256
config.width = 256
config.learning_rate = 5e-3

val_dir = 'test'
train_dir = 'train'

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
    print("Downloading flower dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz", shell=True)

def my_generator(batch_size, img_dir, is_train=False):
    """A generator that returns black and white images and color images"""
    image_filenames = glob.glob(img_dir + "/*")
    counter = 0
    sometimes = lambda aug: iaa.Sometimes(0.25, aug)
    aug = iaa.Sequential([
        sometimes(iaa.CropToFixedSize(config.width, config.height)),
#         iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.5), # vertically flip 20% of all images
#         sometimes(iaa.Sharpen()),
#         sometimes(iaa.ContrastNormalization())
    ])

    while True:
        bw_images = np.zeros((batch_size, config.width, config.height))
        color_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(image_filenames) 
        if ((counter+1)*batch_size>=len(image_filenames)):
            counter = 0
        for i in range(batch_size):
            img = Image.open(image_filenames[counter + i]).resize((config.width, config.height))
            if is_train:
                if min(img.size) < 256:
                    img = img.resize((config.width, config.height))
                _aug = aug.to_deterministic()
                color_images[i] = _aug.augment_image(np.array(img))
                bw_images[i] = _aug.augment_image(np.array(img.convert('L')))
            else:
                img = img.resize((config.width, config.height))
                color_images[i] = np.array(img)
                bw_images[i] = np.array(img.convert('L'))
                
        yield (bw_images, color_images)
        counter += batch_size


# model = Sequential()
# model.add(Reshape((config.height,config.width,1), input_shape=(config.height,config.width)))
# # model.add(BatchNormalization())
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(3, (1, 1), activation='relu', padding='same'))
# # model.add(BatchNormalization(beta_initializer=Constant(value=100.)))

# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(3, (1, 1), activation='relu', padding='same'))

# # Finish model
# model.compile(optimizer='rmsprop',loss='mse')
# 
model, vgg_base = build_model(config.height, config.width, dropout=0.4) 
# model = tf.keras.models.load_model("output/model-best.h5")
# model = create_model(config.height, config.width, num_class=3, dropout=0.5)

def perceptual_distance(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    rmean = ( y_true[:,:,:,0] + y_pred[:,:,:,0] ) / 2;
    r = y_true[:,:,:,0] - y_pred[:,:,:,0]
    g = y_true[:,:,:,1] - y_pred[:,:,:,1]
    b = y_true[:,:,:,2] - y_pred[:,:,:,2]
    
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)));

# optimizer = tf.train.AdamOptimizer(config.learning_rate)
optimizer = RMSprop(lr=config.learning_rate)

model.compile(optimizer=optimizer, loss='mse', metrics=[perceptual_distance])

(val_bw_images, val_color_images) = next(my_generator(145, val_dir, is_train=False))
                    
reduce_lr = ReduceLROnPlateau(monitor='val_perceptual_distance', factor=0.5,
                              patience=10, min_lr=1e-6)

def unfreeze(epoch, logs):
    if epoch > 100:
        for layers in vgg_base.layers:
            layers.trainable = True

model.fit_generator( my_generator(config.batch_size, train_dir, is_train=True),
                     steps_per_epoch=config.num_epochs,
                     epochs=config.num_epochs,
                    callbacks=[
                    reduce_lr,
                    LambdaCallback(on_epoch_end=unfreeze),
                    WandbCallback(monitor="val_perceptual_distance", data_type='image', predictions=100)
                    ],
                     validation_data=(val_bw_images, val_color_images))



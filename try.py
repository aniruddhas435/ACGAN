# import tensorflow as tf

# x = tf.Variable(4.0, trainable = True)
# cost = x**2 - 5 * x + 4
# opt = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for i in range(50):
#     print('loss for itr-{}: {},   value of x: {}'.format(i + 1, sess.run(cost), sess.run(x)))
#     sess.run(opt)

import pandas as pd
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.layers import Dense, Flatten, Conv2D, UpSampling2D, BatchNormalization, Input, LeakyReLU, Activation, Dropout, Reshape, Conv2DTranspose, Embedding
from keras.optimizers import *
from keras.initializers import RandomNormal
from keras.models import Model
from IPython import display
from tqdm import tqdm
import random
from matplotlib import pyplot as plt 


def build_generator():
    channels = 256
    G_input = Input(shape = [100,])

    image_class = Input(shape = [10], dtype = 'float32')
    cl = Dense(units = 100)(image_class)
    G = keras.layers.multiply([G_input, cl])

    print(G.shape)

    G = Dense(units = 7 * 7 * channels)(G)
    G = BatchNormalization(momentum = 0.9)(G)
    G = Activation('relu')(G)
    G = Reshape(target_shape = [7, 7, channels])(G)
    G = Dropout(0.4)(G)

    G = UpSampling2D()(G)
    G = Conv2DTranspose(channels//2, kernel_size = (5, 5), padding = 'same')(G)
    G = BatchNormalization(momentum = 0.9)(G)
    G = Activation('relu')(G)

    G = UpSampling2D()(G)
    G = Conv2DTranspose(channels//4, kernel_size = (5, 5), padding = 'same')(G)
    G = BatchNormalization(momentum = 0.9)(G)
    G = Activation('relu')(G)

    G = Conv2DTranspose(channels//8, kernel_size = (5, 5), padding = 'same')(G)
    G = BatchNormalization(momentum = 0.9)(G)
    G = Activation('relu')(G)

    G = Conv2DTranspose(1, kernel_size = (5, 5), padding = 'same')(G)
    G_output = Activation('sigmoid')(G)
    
    generator = Model([G_input, image_class], G_output)

    generator.summary()
    return generator


def build_discriminator():
    channels = 64
    D_input = Input(shape = [28, 28, 1])

    D = Conv2D(channels, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(D_input)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dropout(0.4)(D)

    D = Conv2D(channels * 2, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dropout(0.4)(D)

    D = Conv2D(channels * 4, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dropout(0.4)(D)

    D = Conv2D(channels * 8, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dropout(0.4)(D)

    D = Flatten()(D)
    D_output1 = Dense(1, activation = 'sigmoid')(D)
    D_output2 = Dense(10, activation = 'softmax')(D)
    discriminator = Model(D_input, [D_output1, D_output2])
    discriminator.summary()

    return discriminator


def build_gan(generator, discriminator):
    noise = Input(shape = [100,])
    image_class = Input(shape = [10], dtype = 'float32')
    image = generator([noise, image_class])
    fake, aux = discriminator(image)
    gan = Model([noise, image_class], [fake, aux])
    gan.summary()
    return gan


if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
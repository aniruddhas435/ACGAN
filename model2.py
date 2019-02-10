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


def load_data():
    X = input_data.read_data_sets('mnist', one_hot = True).train.images
    Y = input_data.read_data_sets('mnist', one_hot = True).train.labels
    print(Y.shape)
    X = X.reshape(X.shape[0], 28, 28, 1).astype(np.float32)
    return X, Y


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


def make_trainable(net, switch = True):
    net.trainable = switch
    for layer in net.layers:
        layer.trainable = switch


def plot_losses(d_losses, g_losses, i):
    plt.figure(figsize = (10, 8))
    plt.plot(d_losses, label = 'discriminative_loss')
    plt.plot(g_losses, label = 'generative_loss')
    plt.savefig('gan_plot.png')
    plt.close()


def plot_gen(generator, i):
    for j in range(10):
        noise = np.random.uniform(-1.0, 1.0, size = [1, 100])
        random_label = np.array([j])
        class_label = keras.utils.to_categorical(random_label, num_classes = 10)
        generated_image = generator.predict([noise, class_label])
        plt.figure()
        plt.imshow(generated_image[0, :, :, 0])
        plt.savefig('generated_images\itr{}_digit-{}.png'.format(i, random_label[0]))
        # print(j)
        plt.close()


def train(X, Y, generator, discriminator, GAN, epochs = 1000, batch_size = 32):
    d_losses = []
    g_losses = []

    for i in tqdm(range(1001, epochs)):
        indices = np.random.randint(0, X.shape[0], size = batch_size)
        real_images = X[indices, :, :, :]
        real_labels = Y[indices, :]

        noise = np.random.uniform(-1.0, 1.0, size = [batch_size, 100])
        random_labels = np.random.randint(0, 10, size = [batch_size])
        random_labels = keras.utils.to_categorical(random_labels, num_classes = 10)

        generated_images = generator.predict([noise, random_labels])
        
        X_train = np.concatenate((real_images, generated_images))
        real_or_fake = np.array([1] * batch_size + [0] * batch_size)
        class_labels = np.concatenate((real_labels, random_labels))

        make_trainable(discriminator)
        for t in range(2):
            d_loss_total, d_loss_source, d_loss_class = discriminator.train_on_batch(X_train, [real_or_fake, class_labels])
        
        noise = np.random.uniform(-1.0, 1.0, size = [batch_size * 2, 100])
        random_labels = np.random.randint(0, 10, size = [batch_size * 2])
        class_labels = keras.utils.to_categorical(random_labels, 10)
        real_or_fake = np.array([1] * batch_size * 2)

        make_trainable(discriminator, switch = False)
        g_loss_total, g_loss_source, g_loss_class = GAN.train_on_batch([noise, class_labels], [real_or_fake, class_labels])

        d_losses.append(d_loss_total)
        g_losses.append(g_loss_total)

        if(i > 1300): 
            d_losses.pop(0)
            g_losses.pop(0)
        
        if i % 25 == 0:
            print('\ndiscriminator:\nJ_total: {}, J_source: {}, J_class: {}'.format(d_loss_total, d_loss_source, d_loss_class))
            print('generator:\nJ_total: {}, J_source: {}, J_class: {}\n'.format(g_loss_total, g_loss_source, g_loss_class))
            plot_losses(d_losses, g_losses, i)
            plot_gen(generator, i)

        if i % 100 == 0:
            generator.save('generator\model-{:04d}.h5'.format(i))
            discriminator.save('discriminator\model-{:04d}.h5'.format(i))

    print('\ndiscriminator:\nJ_total: {}, J_source: {}, J_class: {}'.format(d_loss_total, d_loss_source, d_loss_class))
    print('generator:\nJ_total: {}, J_source: {}, J_class: {}\n'.format(g_loss_total, g_loss_source, g_loss_class))
    plot_losses(d_losses, g_losses, epochs)
    plot_gen(generator, epochs)

    generator.save('generator\model.h5')
    discriminator.save('discriminator\model.h5')


if __name__ == '__main__':
    train_images, train_labels = load_data()
    X = train_images
    Y = train_labels
    print(X.shape)
    print(Y.shape)

    generator = build_generator()

    discriminator = build_discriminator()
    discriminator.compile(loss = ['binary_crossentropy', 'categorical_crossentropy'], optimizer = RMSprop(lr = 0.0002, decay = 6e-8))

    make_trainable(discriminator, switch = False)
    GAN = build_gan(generator, discriminator)
    GAN.compile(loss = ['binary_crossentropy', 'categorical_crossentropy'], optimizer = RMSprop(lr=0.0001, decay = 3e-8))

    train(X, Y, generator, discriminator, GAN, epochs = 5000, batch_size = 256)
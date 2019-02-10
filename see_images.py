import pandas as pd
import cv2
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv2D, UpSampling2D, BatchNormalization, Input, LeakyReLU, Activation, Dropout, Reshape, Conv2DTranspose
from keras.optimizers import *
from keras.initializers import RandomNormal
from keras.models import Model
from IPython import display
from tqdm import tqdm
import random
from matplotlib import pyplot as plt

generator = keras.models.load_model('generator\model.h5')
# discriminator = keras.models.load_model('discriminator\model-4000.h5')

noise = np.random.uniform(-1.0, 1.0, size = [1, 100])
for j in range(9):
    for i in range(10):
        digit = np.zeros((1, 10))
        digit[:, j] = (10 - i) * 0.1
        digit[:, j + 1] = i * 0.1
        image = generator.predict([noise, digit])
        plt.imshow(image[0, :, :, 0])
        plt.savefig('image_transition\{}_to_{}_stage_{}.png'.format(j, j + 1, i))
        plt.show()
        plt.close()
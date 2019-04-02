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
discriminator = keras.models.load_model('discriminator\model-4000.h5')

string = input('Enter the digits with spaces in between: ')
digits = string.split(' ')

count = 0
images = []

noise = np.random.uniform(-1.0, 1.0, size = [1, 100])
for i in tqdm(range(len(digits) - 1)):
    for j in range(10):
        digit = np.zeros((1, 10))
        digit[:, int(digits[i])] = (10 - j) * 0.1
        digit[:, int(digits[i + 1])] = j * 0.1
        image = generator.predict([noise, digit])
        plt.imshow(image[0, :, :, 0])
        plt.savefig('image.png'.format(count))
        images.append(cv2.imread('image.png'))
        plt.close()
        count += 1

height, width, layer = images[0].shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('twining_digits.avi', fourcc, 4.0, (width, height))

for i in range(len(images)):
    video.write(images[i])

cv2.destroyAllWindows()
video.release()

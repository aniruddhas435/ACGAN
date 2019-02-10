import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


generator = load_model('generator\model-4000.h5')

def write_number(number):
    noise = np.random.uniform(-1.0, 1.0, size = [1, 100])
    n = len(number)
    image = generator.predict([noise, np.array([int(number[0])])])

    for i in range(1, n):
        noise = np.random.uniform(-1.0, 1.0, size = [1, 100])
        curr_digit = int(number[i])
        curr_image = generator.predict([noise, np.array([curr_digit])])
        image = np.concatenate((image, curr_image), axis = 2)
    
    plt.imshow(image[0, :, :, 0], cmap = 'gray')
    plt.show()
    plt.close()


while True:
    write_number(str(input('Enter the number: ')))
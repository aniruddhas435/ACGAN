import numpy as np
import cv2

images = []
for j in range(9):
    for i in range(10):
        image = cv2.imread('image_transition\{}_to_{}_stage_{}.png'.format(j, j + 1, i), cv2.IMREAD_COLOR)
        images.append(image)

height, width, layer = images[0].shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('writing_numbers.avi', fourcc, 4.0, (width, height))

for i in range(len(images)):
    video.write(images[i])

cv2.destroyAllWindows()
video.release()
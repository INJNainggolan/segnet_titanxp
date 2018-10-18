
import numpy as np
import cv2
image = np.zeros((100,100,3),dtype=np.uint8)
print(image)
print(image.shape)
print(image.shape[0])
print(image.shape[1])
print(image.shape[2])
print(image[99][99][0])

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j][0] == 0:               #像素值为0
            image[i][j][0] = 57
            image[i][j][1] = 104
            image[i][j][2] = 205

cv2.imwrite('red.png',image[:,:])





import cv2
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
if __name__=='__main__':
    for a in range(0, 24):
        rgbPic = cv2.imread('/home/cqnu/dataset/Potsdam/Potsdam_num/label_III/{0}.png'.format(a))
        grayPic = rgb2gray(rgbPic)
        cv2.imwrite('/home/cqnu/dataset/Potsdam/Potsdam_num/label_gray/{0}.png'.format(a), grayPic[:, :])





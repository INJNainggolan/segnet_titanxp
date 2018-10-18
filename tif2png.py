import cv2

if __name__=='__main__':
    for a in range(0, 24):
        image = cv2.imread('/home/cqnu/dataset/Potsdam/Potsdam_num/src_tif/{0}.tif'.format(a))
        cv2.imwrite('/home/cqnu/dataset/Potsdam/Potsdam_num/src/{0}.png'.format(a), image[:, :])
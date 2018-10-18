import cv2

if __name__=='__main__':
    for a in range(0, 24):
        image = cv2.imread('/home/cqnu/dataset/Potsdam/Potsdam_num/label_RGB/{0}.tif'.format(a))

        print(image)

        print(image.shape)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0] == 255 and image[i][j][1] == 255 and image[i][j][2] == 255:
                    image[i][j][0] = 1
                    image[i][j][1] = 1
                    image[i][j][2] = 1
                elif image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 255:
                    image[i][j][0] = 2
                    image[i][j][1] = 2
                    image[i][j][2] = 2
                elif image[i][j][0] == 0 and image[i][j][1] == 255 and image[i][j][2] == 255:
                    image[i][j][0] = 3
                    image[i][j][1] = 3
                    image[i][j][2] = 3
                elif image[i][j][0] == 0 and image[i][j][1] == 255 and image[i][j][2] == 0:
                    image[i][j][0] = 4
                    image[i][j][1] = 4
                    image[i][j][2] = 4
                elif image[i][j][0] == 255 and image[i][j][1] == 255 and image[i][j][2] == 0:
                    image[i][j][0] = 0
                    image[i][j][1] = 0
                    image[i][j][2] = 0
                elif image[i][j][0] == 255 and image[i][j][1] == 0 and image[i][j][2] == 0:
                    image[i][j][0] = 0
                    image[i][j][1] = 0
                    image[i][j][2] = 0
                else:
                    image[i][j][0] = 0
                    image[i][j][1] = 0
                    image[i][j][2] = 0

        cv2.imwrite('/home/cqnu/dataset/Potsdam/Potsdam_num/label_III/{0}.png'.format(a),image[:,:])
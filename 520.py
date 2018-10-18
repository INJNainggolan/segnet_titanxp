import cv2

if __name__ == '__main__':
    for a in range(3, 8):
        image = cv2.imread('/home/zq/dataset/dataset_RSI_eCognition/train_all/label_before/{0}.tif'.format(a), cv2.IMREAD_GRAYSCALE)

        print(image)

        print(image.shape)

        #####################################
        # 其他0 黑色 0 0 0 | 0 0 0

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 5:  # 修改other为0
                    image[i, j] = 0

        #####################################
        cv2.imwrite('/home/zq/dataset/dataset_RSI_eCognition/train_all/label/{0}.tif'.format(a), image[:, :])
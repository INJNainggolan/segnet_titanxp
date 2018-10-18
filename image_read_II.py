import cv2

#image = cv2.imread('G:\\xuyue\\remote_sensing_image\\output_segnet_predict\\predict2.png')
image = cv2.imread('/home/user/PycharmProjects/pycharm_workspace/xuyue/remote_sensing_image/output_segnet_predict/predict5.png' )

print(image)

print(image.shape)

'''
for i in range(image.shape[2]):
    for j in range(image.shape[1]):
        for k in range(image.shape[0]):
            if image[k][j][i] == 2:
                image[k][j][i] = image[k][j][i] + 200
            if image[k][j][i] == 1:
                image[k][j][i] = image[k][j][i] + 125
            if image[k][j][i] == 0:
                image[k][j][i] = image[k][j][i] + 50
'''
#####################################
#其他0 黑色 0 0 0 | 0 0 0

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j][0] == 1:               #植被1 绿色 0 255 0 | 0 255 0
            image[i][j][0] = 5
            image[i][j][1] = 255
            image[i][j][2] = 5

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j][0] == 2:             #建筑3 黄色 255 255 0 | 0 255 255 |
            image[i][j][0] = 5
            image[i][j][1] = 255
            image[i][j][2] = 255

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j][0] == 3:             #水体4 蓝色 0 0 255 | 255 0 0 |
            image[i][j][0] = 255
            image[i][j][1] = 5
            image[i][j][2] = 5

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j][0] == 4:             #道路2 棕色 139 35 35 | 35 35 139 |
            image[i][j][0] = 57
            image[i][j][1] = 104
            image[i][j][2] = 205

#####################################
#cv2.imwrite('G:/xuyue/remote_sensing_image/output_segnet_predict/predict2_color.png',image[:,:])
cv2.imwrite('/home/user/PycharmProjects/pycharm_workspace/xuyue/remote_sensing_image/output_segnet_predict/predict5_color_II.png',image[:,:])
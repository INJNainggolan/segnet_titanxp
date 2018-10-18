import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K
K.clear_session()

#TEST_SET = ['1.png','2.png','3.png','4.png','5.png']
TEST_SET = ['1.tif','2.tif','3.tif','4.tif','5.tif','6.tif','7.tif','8.tif','9.tif','10.tif']

image_size = 256

classes = [0. ,  1.,  2.,   3.  , 4.]
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 


def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
#    ap.add_argument("-s", "--stride", required=False,
#       help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args



    
def predict(args):
#def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])

#    stride = args['stride']
    stride = 128

    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('/home/zq/dataset/dataset_RSI_eCognition/test/' + path)
        h,w,_ = image.shape
        print("image shape =",image.shape)

        padding_h = (h//stride + 1) * stride
        print("padding_h=",padding_h)

        padding_w = (w//stride + 1) * stride
        print("padding_w=", padding_w)

        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        print("padding_img.shape",padding_img.shape)

        padding_img[0:h,0:w,:] = image[:,:,:]
        print("padding_img.shape=",padding_img.shape)
        print("padding_img",padding_img)

        padding_img = padding_img.astype("float") / 255.0
        print("padding_img.shape=", padding_img.shape)
        print("padding_img", padding_img)

        padding_img = img_to_array(padding_img)
        print("padding_img.shape=", padding_img.shape)
        print("padding_img", padding_img)

        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        print("mask_whole.shape=",mask_whole.shape)

        print("image size=",image_size)
        print("stride=",stride)
        print("padding_h//stride=",padding_h//stride)
        print("padding_w//stride=", padding_w // stride)

        print("******************************************************")

        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
                print("i,j",i,j)
                print("crop.shape=",crop.shape)
                ch,cw,_ = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue

                crop = np.expand_dims(crop, axis=0)
                print('crop:',crop.shape)
                pred = model.predict_classes(crop,verbose=2)
                pred = labelencoder.inverse_transform(pred[0])  
                print(np.unique(pred))
                pred = pred.reshape((256,256)).astype(np.uint8)
                print('pred:',pred.shape)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]


        
        cv2.imwrite('/home/zq/output/segnet_output_tl_ft/7th/predict_III'+str(n+1)+'.png',mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)
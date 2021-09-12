import glob
import os
import time
import argparse
import numpy as np
from PIL import Image
from libtiff import TIFF
import tensorflow as tf
import cv2
import keras
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from src.model import coarse_generator,fine_generator


global g_local_model
global g_global_model


def normalize_pred(img,mask):
    img = np.reshape(img,[1,128,128,3])
    mask = np.reshape(mask,[1,128,128,1])
    img_coarse = tf.image.resize(img, (64,64), method=tf.image.ResizeMethod.LANCZOS3)
    img_coarse = (img_coarse - 127.5) / 127.5
    img_coarse = np.array(img_coarse)
    mask_coarse = tf.image.resize(mask, (64,64), method=tf.image.ResizeMethod.LANCZOS3)
    mask_coarse = (mask_coarse - 127.5) / 127.5
    mask_coarse = np.array(mask_coarse)
    start_time = time.time()
    X_fakeB_coarse,x_global = g_global_model.predict([img_coarse,mask_coarse])
    
    X_fakeB_coarse = (X_fakeB_coarse + 1) /2.0
    X_fakeB_coarse = cv2.normalize(X_fakeB_coarse, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    X_fakeB_coarse = tf.image.resize(X_fakeB_coarse, (128,128), method=tf.image.ResizeMethod.LANCZOS3)
    pred_img_coarse = X_fakeB_coarse[:,:,:,0]
    img = (img - 127.5) / 127.5
    mask = (mask - 127.5) / 127.5
    X_fakeB = g_local_model.predict([img,mask,x_global])
    end_time = time.time()
    print(end_time-start_time)
    X_fakeB = (X_fakeB + 1) /2.0
    X_fakeB = cv2.normalize(X_fakeB, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    pred_img = X_fakeB[:,:,:,0]
    return np.asarray(pred_img,dtype=np.float32),np.asarray(pred_img_coarse,dtype=np.float32)

def strided_crop(img, mask, img_h,img_w,height, width,stride=1):
    full_prob = np.zeros((img_h, img_w),dtype=np.float32)
    full_prob_coarse = np.zeros((img_h, img_w),dtype=np.float32)
    full_sum = np.ones((img_h, img_w),dtype=np.float32)
    full_sum_coarse = np.ones((img_h, img_w),dtype=np.float32)
    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_mask_arr = mask[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                pred,pred_coarse = normalize_pred(crop_img_arr,crop_mask_arr)
                full_prob[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred[0]
                full_sum[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
                full_prob_coarse[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred_coarse[0]
                full_sum_coarse[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
                i = i + 1
    out_img = full_prob / full_sum
    out_img_coarse = full_prob_coarse / full_sum_coarse
    return out_img, out_img_coarse




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='DRIVE', required=True, choices=['DRIVE','CHASE','STARE'])
    parser.add_argument('--out_dir', type=str, default='pred', required=False)
    parser.add_argument('--weight_name_global',type=str, help='path/to/global/weight/.h5 file', required=True)
    parser.add_argument('--weight_name_local',type=str, help='path/to/local/weight/.h5 file', required=True)
    parser.add_argument('--stride', type=int, default=3, help='For faster inference use stride 16/32, for better result use stride 3.')
    parser.add_argument('--out_dir')
    args = parser.parse_args() 

## Input dimensions

image_shape_fine = (128,128,3)
mask_shape_fine = (128,128,1)
label_shape_fine = (128,128,1)
image_shape_x_coarse = (64,64,128)
image_shape_coarse = (64,64,3)
mask_shape_coarse = (64,64,1)
label_shape_coarse = (64,64,1)
img_shape_g = (64,64,3)
ndf=64
ncf=128
nff=128

## Load models
K.clear_session()
opt = Adam()
g_local_model = fine_generator(x_coarse_shape=image_shape_x_coarse,input_shape=image_shape_fine,mask_shape=mask_shape_fine,nff=nff)
g_local_model.load_weights(args.weight_name_local)
g_local_model.compile(loss='mse', optimizer=opt)
g_global_model = coarse_generator(img_shape=image_shape_coarse,mask_shape=mask_shape_coarse,ncf=ncf)
g_global_model.load_weights(args.weight_name_global)
g_global_model.compile(loss='mse',optimizer=opt)


## Create Output Directory
out_path = args.out_dir
directories = [out_path,out_path+'/Coarse',out_path+'/Fine']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

## Find file numbers,paths or names
if args.test_data == 'DRIVE':
    limit = 20
elif args.test_data == 'CHASE':
    filenames = glob.glob("CHASE/test/images/*.jpg")
    limit = len(filenames)

elif args.test_data == 'STARE':
    arr = ["im0001","im0082","im0236","im0319"]
    limit = 4


## Iterating for each image

for i in range(0,limit):

    if args.test_data == 'DRIVE':
        
        if i<9:
            ind = str(0)
        else:
            ind = str("")
        label_name = "DRIVE/test/1st_manual/"+ind+str(i+1)+"_manual1.gif"
        label = Image.open(label_name)
        label_arr = np.asarray(label,dtype=np.float32)
        img_name = "DRIVE/test/images/"+ind+str(i+1)+"_test.tif"
        tif = TIFF.open(img_name)
        img_arr = tif.read_image(tif)
        img_arr = np.asarray(img_arr,dtype=np.float32)
        mask_name = "DRIVE/test/mask/"+ind+str(i+1)+"_test_mask.gif"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask,dtype=np.float32)

    elif args.test_data == 'CHASE':
        k = filenames[i].split('/')
        k = k[-1].split('.')[0]
        label_name = "CHASE/test/labels/"+k+"_1stHO.png"
        label = Image.open(label_name)
        label_arr = np.asarray(label,dtype=np.float32)
        img_name = "CHASE/test/images/"+k+".jpg"
        img = Image.open(img_name)
        img_arr = np.asarray(img,dtype=np.float32)
        mask_name = "CHASE/test/mask/"+k+"_mask.png"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask,dtype=np.float32)

    elif args.test_data == 'STARE':
        label_name = "STARE/test/labels-ah/"+arr[i]+".ah.ppm"
        label = Image.open(label_name)
        label_arr = np.asarray(label,dtype=np.float32)
        img_name = "STARE/test/stare-original-images/"+arr[i]+".ppm"
        img = Image.open(img_name)
        img_arr = np.asarray(img,dtype=np.float32)
        mask_name = "STARE/test/mask/"+arr[i]+"_mask.png"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask,dtype=np.float32)


    ## Get the output predictions as array


    ## Stride =3 (best result),  Stride = 32 (faster result).
    out_img,out_img_coarse = strided_crop(img_arr, mask_arr, mask_arr.shape[0], mask_arr.shape[1], 128, 128,args.stride)

    out_img[mask_arr==0] = 0
    out_img[out_img>=0.5] = 1
    out_img[out_img<0.5] = 0
    save_im = out_img.astype(np.uint8)
    save_im[save_im==1] = 255
    save_im = Image.fromarray(save_im)


    out_img_coarse[mask_arr==0] = 0
    out_img_coarse[out_img_coarse>=0.5] = 1
    out_img_coarse[out_img_coarse<0.5] = 0
    save_im_coarse = out_img_coarse.astype(np.uint8)
    save_im_coarse[save_im_coarse==1] = 255
    save_im_coarse = Image.fromarray(save_im_coarse)
    
    ## Save files

    if args.test_data == 'DRIVE':

        pred_name = directories[2]+"/"+ind+str(i+1)+".png"
        pred_name_coarse = directories[1]+"/"+ind+str(i+1)+"coarse.png"
        save_im.save(pred_name)
        save_im_coarse.save(pred_name_coarse)

    elif args.test_data == 'CHASE':
        pred_name = directories[2]+"/"+k+".png"
        pred_name_coarse = directories[1]+"/"+k+"coarse.png"
        save_im.save(pred_name)
        save_im_coarse.save(pred_name_coarse)

    elif args.test_data == 'STARE':
        pred_name = directories[2]+"/"+arr[i]+".png"
        pred_name_coarse = directories[1]+"/"+arr[i]+"coarse.png"
        save_im.save(pred_name)
        save_im_coarse.save(pred_name_coarse)

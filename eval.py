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
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,f1_score,roc_auc_score,auc,recall_score, auc,roc_curve


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
    
    _,x_global = g_global_model.predict([img_coarse,mask_coarse])

    img = (img - 127.5) / 127.5
    mask = (mask - 127.5) / 127.5
    X_fakeB = g_local_model.predict([img,mask,x_global])
    X_fakeB = (X_fakeB + 1) /2.0
    X_fakeB = cv2.normalize(X_fakeB, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    pred_img = X_fakeB[:,:,:,0]
    return np.asarray(pred_img,dtype=np.float32)


def strided_crop(img, mask, img_h,img_w,height, width,stride=1):
  
    full_prob = np.zeros((img_h, img_w),dtype=np.float32)
    full_sum = np.ones((img_h, img_w),dtype=np.float32)
    
    max_x = int(((img.shape[0]-height)/stride)+1)
    #print("max_x:",max_x)
    max_y = int(((img.shape[1]-width)/stride)+1)
    #print("max_y:",max_y)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_mask_arr = mask[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                pred = normalize_pred(crop_img_arr,crop_mask_arr)
                crop_img_arr 
                full_prob[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred[0]
                full_sum[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
                i = i + 1
                #print(i)
    out_img = full_prob / full_sum
    return out_img




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='DRIVE', required=True, choices=['DRIVE','CHASE','STARE'])
    parser.add_argument('--weight_name_global',type=str, help='path/to/global/weight/.h5 file', required=True)
    parser.add_argument('--weight_name_local',type=str, help='path/to/local/weight/.h5 file', required=True)
    parser.add_argument('--stride', type=int, default=3, help='For faster inference use stride 16/32, for better result use stride 3.')
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


y_true = np.zeros((limit,960,999),dtype=np.float32)
y_pred = np.zeros((limit,960,999),dtype=np.float32)
y_pred_auc = np.zeros((limit,960,999),dtype=np.float32)
c = 0
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
    out_img = strided_crop(img_arr, mask_arr, mask_arr.shape[0], mask_arr.shape[1], 128, 128,args.stride)

    out_img[mask_arr==0] = 0
    y_pred_auc[c,:,:] = out_img
    out_img[out_img>=0.5] = 1
    out_img[out_img<0.5] = 0
    y_true[c,:,:] = label_arr 
    y_pred[c,:,:] = out_img
    c = c +1
    
y_true = y_true.flatten()
y_pred = y_pred.flatten()
y_pred_auc = y_pred_auc.flatten()
confusion = confusion_matrix(y_true, y_pred)
print(confusion)

tn, fp, fn, tp = confusion.ravel()   
metric_cal = time.time()
if float(np.sum(confusion)) != 0:
    accuracy =  float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))
specificity = 0
if float(confusion[0, 0] + confusion[0, 1]) != 0:
    specificity = tn / (tn + fp)
print("Specificity: " + str(specificity))
sensitivity = 0
if float(confusion[1, 1] + confusion[1, 0]) != 0:
    sensitivity = tp / (tp + fn) 
print("Sensitivity: " + str(sensitivity))

precision = 0
if float(confusion[1, 1] + confusion[0, 1]) != 0:
    precision = tp / (tp + fp) 
print("Precision: " + str(precision))


F1_score = 2*tp/(2*tp+fn+fp) 
print("F1 score (F-measure): " + str(F1_score))

AUC_ROC = roc_auc_score(y_true, y_pred_auc)
print("AUC_ROC: " + str(AUC_ROC))

ssim = ssim(y_true, y_pred, data_range=y_true.max()-y_true.min())
print("SSIM: " + str(ssim))

meanIOU = jaccard_similarity_score(y_true,y_pred,normalize=True)
print("meanIOU: " + str(meanIOU))


from PIL import Image
import numpy as np
import random
import os
import argparse

def strided_crop(img, mask, label, height, width, name,stride=1):
    directories = ['Drive_crop','Drive_crop/Images','Drive_crop/Masks','Drive_crop/labels']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    max_x = int(((img.shape[0]-height)/stride)+1)
    #print("max_x:",max_x)
    max_y = int(((img.shape[1]-width)/stride)+1)
    #print("max_y:",max_y)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                #print(crop_img_arr.shape)
                crop_mask_arr = mask[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_label_arr = label[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_img = Image.fromarray(crop_img_arr)
                crop_mask = Image.fromarray(crop_mask_arr)
                crop_label = Image.fromarray(crop_label_arr)
                img_name = directories[1] + "/" + name + "_" + str(i+1)+".png"
                mask_name = directories[2] + "/" + name + "_mask_" + str(i+1)+".png"
                label_name = directories[3] + "/" + name + "_label_" + str(i+1)+".png"
                crop_img.save(img_name)
                crop_mask.save(mask_name)
                crop_label.save(label_name)
                i = i + 1
                #print(i)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--stride', type=int, default=32)
    args = parser.parse_args()

    for i in range(21,41):
            
        img_name = "DRIVE/training/images/"+str(i)+"_training.tif"
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = "DRIVE/training/mask/"+str(i)+"_training_mask.gif"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        label_name = "DRIVE/training/1st_manual/"+str(i)+"_manual1.gif"
        label = Image.open(label_name)
        label_arr = np.asarray(label)
        
        name = str(i)
        strided_crop(img_arr, mask_arr, label_arr, args.input_dim, args.input_dim,name,args.stride)
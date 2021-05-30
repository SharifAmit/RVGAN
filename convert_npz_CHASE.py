import numpy as np
from numpy import asarray,savez_compressed
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse
import glob


#load all images in a directory into memory
def load_images(imgpath,maskpath,labelpath,n_crops, size=(128,128)):
    src_list, mask_list, label_list = list(), list(), list()
    img_list = glob.glob("Chase_db1/training/images/*.jpg")
    for i in img_list:
        i = i.split('\\')
        i = i[1].split('.')
        #print(i[0])
        for j in range(n_crops):
            # load and resize the image
            filename = i[0]+"_"+str(j+1)+".png"
            mask_name = i[0]+"_mask_" + str(j+1)+".png"
            label_name = i[0]+"_label_" + str(j+1)+".png"
            
            img = load_img(imgpath + filename, target_size=size)
            fundus_img = img_to_array(img)

            mask = load_img(maskpath + mask_name, target_size=size,color_mode="grayscale")
            mask_img = img_to_array(mask)
            
            label = load_img(labelpath + label_name, target_size=size,color_mode="grayscale")
            label_img = img_to_array(label)
            
            # split into satellite and map
            src_list.append(fundus_img)
            mask_list.append(mask_img)
            label_list.append(label_img)
    return [asarray(src_list), asarray(mask_list), asarray(label_list)]
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=(128,128))
    parser.add_argument('--n_crops', type=int, default=756)
    parser.add_argument('--outfile_name', type=str, default='CHASE')
    args = parser.parse_args() 

    # dataset path
    imgpath = 'Chase_crop/Images/'
    maskpath = 'Chase_crop/Masks/'
    labelpath = 'Chase_crop/labels/'
    # load dataset
    [src_images, mask_images, label_images] = load_images(imgpath,maskpath,labelpath,args.n_crops,args.input_dim)
    print('Loaded: ', src_images.shape, mask_images.shape, label_images.shape)
    # save as compressed numpy array
    filename = args.outfile_name+'.npz'
    savez_compressed(filename, src_images, mask_images, label_images)
    print('Saved dataset: ', filename)
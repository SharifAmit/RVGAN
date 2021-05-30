from PIL import Image
import numpy as np
import os
import argparse
import glob

def strided_crop(img, mask, label, height, width,name,stride=1):
    directories = ['Stare_crop','Stare_crop/Images','Stare_crop/Masks','Stare_crop/labels']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    max_x = int(((img_arr.shape[0]-height)/stride)+1)
    #print("max_x:",max_x)
    max_y = int(((img_arr.shape[1]-width)/stride)+1)
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


    # Creating masks

    train_test = ['training','test']
    for n in train_test: 
        img_dir = "Stare/"+n+"/stare-original-images/*.ppm"
        images = glob.glob(img_dir)
        directory = 'Stare/'+n+'/mask'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in images:
            image_name = i.split('\\')[1].split('.')[0]
            #print(image_name)
            im = Image.open(i)
            im_gray = im.convert('L')
            np_im = np.array(im_gray)

            np_mask = np.zeros((np_im.shape[0],np_im.shape[1]))
            np_mask[np_im[:,:] >40] = 255
            mask = Image.fromarray(np_mask)
            mask = mask.convert('L')
            mask_name = directory + "/" + image_name + "_mask.png"
            mask.save(mask_name)

    # Crop from images
    images = glob.glob("Stare/training/stare-original-images/*.ppm")
    for i in images:
        i = i.split('\\')
        i = i[1].split('.')
        img_name = "Stare/training/stare-original-images/"+i[0]+'.ppm'
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = "Stare/training/mask/"+i[0]+"_mask.png"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        label_name = "Stare/training/labels-ah/"+i[0]+'.ah.ppm'
        label = Image.open(label_name)
        label_arr = np.asarray(label)
        
        name = i[0]
        strided_crop(img_arr, mask_arr, label_arr, args.input_dim, args.input_dim,name,args.stride)

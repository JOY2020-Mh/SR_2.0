import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imageio
import numpy as np
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


crop_size = 500
scale_factor = 4
GT_dir = '/home/miaohuan/Downloads/Steven_new_dataset/T1-1/HR'
LR_dir = '/home/miaohuan/Downloads/Steven_new_dataset/T1-1/LR'
GT_image_filenames = []
LR_image_filenames = []
for image_name in sorted(glob.glob(GT_dir+'/*')):
    GT_image_filenames.append(image_name)
for image_name in sorted(glob.glob(LR_dir+'/*')):
    LR_image_filenames.append(image_name)

num = len(GT_image_filenames)
step = 0
for epoch in (range(12)):
    for index in (range(num)): 
        GT_img = img = Image.open(GT_image_filenames[index])
        LR_img = img = Image.open(LR_image_filenames[index])

        lr_h = LR_img.size[1] 
        lr_w = LR_img.size[0]

        print(GT_image_filenames[index])
        hr_crop_w = crop_size
        hr_crop_h = crop_size

        # determine LR crop image size
        lr_crop_w = hr_crop_w // scale_factor
        lr_crop_h = hr_crop_h // scale_factor


         #ramdom crop image
        hr_w = GT_img.size[0]
        hr_h = GT_img.size[1]

        #hr_box = (int(hr_w*0.1), int(hr_h *0.1), int(hr_w * 0.9), int(hr_h * 0.9)) 
        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        img_LR = LR_img.crop((rnd_w, rnd_h, rnd_w + lr_crop_w, rnd_h + lr_crop_h))
        rnd_h_HR, rnd_w_HR = int(rnd_h * scale_factor), int(rnd_w * scale_factor)
        img_HR = GT_img.crop((rnd_w_HR, rnd_h_HR, rnd_w_HR + hr_crop_w, rnd_h_HR + hr_crop_h))


        p1 = random.random()
        if p1 > 0.5:
            transform = transforms.RandomHorizontalFlip(p=1)
            img_HR = transform(img_HR)
            img_LR = transform(img_LR)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(p=1)
            img_HR = transform(img_HR)
            img_LR = transform(img_LR)
       
        p2 = random.random()
        if p2 >= 2/3:
            img_HR.transpose(Image.ROTATE_90)
            img_LR.transpose(Image.ROTATE_90)
        elif 2/3 > p2 >= 1/3:
            img_HR.transpose(Image.ROTATE_180)
            img_LR.transpose(Image.ROTATE_180)
        else:
            img_HR = img_HR
            img_LR = img_LR

        new_HR_dir = './1125_image_augument/HR'
        new_LR_dir = './1125_image_augument/LR'
        HR_path = GT_image_filenames[index]
        im_name = os.path.split(HR_path)[-1].split('.png')[0]

        # print(os.path.split(HR_path)[-1])
        # print(im_name)
        # print(im_name.split('_'))
        # new_HR_name = im_name.split('.')[0] + '_' + im_name.split('_')[4] + '_' + str(index) +'_Visidon.jpg'
        new_HR_name = im_name.split('.yuv')[0] + '_' + str(step) + '_HR''.png'
        print(new_HR_name)
        new_HR_path = os.path.join(new_HR_dir,new_HR_name)
        imageio.imsave(new_HR_path, img_HR)

        LR_path = LR_image_filenames[index]
        lr_image_name = os.path.split(LR_path)[-1].split('.png')[0]
        new_LR_name = lr_image_name.split('.yuv')[0] + '_' + str(step) + '_LR''.png'
        print(new_LR_name)
        new_LR_path = os.path.join(new_LR_dir,new_LR_name)
        imageio.imsave(new_LR_path, img_LR)
        step = step + 1

        # imageio.imsave('0_cropped_flip_rotate_1113_test111/img_LR'+str(index)+'.png', img_LR)
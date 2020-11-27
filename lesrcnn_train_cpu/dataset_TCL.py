import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import *
from os import listdir
from os.path import join

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1] #h,w,channel [:-1] beside the final element, such as channel 
    x = random.randint(0, w-size) #random number
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy() #low-resolution patch
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()#high-resolution patch

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1) #left and right move
        im2 = np.flipud(im2)

    if random.random() < 0.5: #up and down move
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3]) #rote
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, GT_dir, LR_dir, crop_size=128, scale_factor=4):
        super(TrainDatasetFromFolder, self).__init__()
        self.GT_image_filenames = []
        self.LR_image_filenames = []
        for image_name in sorted(glob.glob(GT_dir+'/*')):
            self.GT_image_filenames.append(image_name)
        for image_name in sorted(glob.glob(LR_dir+'/*')):
            self.LR_image_filenames.append(image_name)

        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        GT_img = Image.open(self.GT_image_filenames[index])
        LR_img = Image.open(self.LR_image_filenames[index])
        #print("GT_size is:", GT_img.size)
        #resize LR image to the needed size
        lr_h = LR_img.size[1]
        lr_w = LR_img.size[0]
        #transform = Resize((lr_h, lr_w), interpolation=Image.BILINEAR)
        #LR_img = transform(LR_img)
        # print self.GT_image_filenames[index]+ 'GT,' + self.LR_image_filenames[index] + 'LR'
        # determine valid HR image size with scale factor
        # self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_crop_w = self.crop_size
        hr_crop_h = self.crop_size

        # determine LR crop image size
        lr_crop_w = hr_crop_w // self.scale_factor
        lr_crop_h = hr_crop_h // self.scale_factor

        # center crop
        #rnd_h = (lr_h - lr_crop_h)/2
        #rnd_w = (lr_w - lr_crop_w)/2
        #ramdom crop image
        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        img_LR = LR_img.crop((rnd_h, rnd_w, rnd_h + lr_crop_h, rnd_w + lr_crop_w))
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale_factor), int(rnd_w * self.scale_factor)
        img_HR = GT_img.crop((rnd_h_HR, rnd_w_HR, rnd_h_HR + hr_crop_h, rnd_w_HR + hr_crop_w))
        #print(img_HR.size)
        #print(img_LR.size)
       # print('img_HR.size()', img_HR.size)
        #print('img_LR.size', img_LR.size)
        ###random flip 
        #if random.random() < 0.5:
        #  img_LR = np.flipud(img_LR) #left and right move
        #  img_HR = np.flipud(img_HR)

        #if random.random() < 0.5: #up and down move
        #  img_LR = np.fliplr(img_LR)
        #  img_HR = np.fliplr(img_HR)

        #angle = random.choice([0, 1, 2, 3]) #rote
        #img_LR = np.rot90(img_LR, angle)
        #img_HR = np.rot90(img_HR, angle)

        img_HR=ToTensor()(img_HR)
        img_LR=ToTensor()(img_LR)

        return {'img_LR': img_LR, 'img_HR': img_HR}
        ### croped img_HR and img_LR

    def __len__(self):
        return len(self.GT_image_filenames)


class TestDataset(data.Dataset):
    def __init__(self, test_GT_dir, test_LR_dir, scale):
        super(TestDataset, self).__init__()

        self.scale = scale
        self.lr = [join(test_LR_dir, x) for x in sorted(listdir(test_LR_dir))]
        self.hr = [join(test_GT_dir, x) for x in sorted(listdir(test_GT_dir))]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])
        '''
        #This is only used to test code, which makes me more know the code. 
        ss = np.asarray(hr) (644,1024,3)
        print ss.shape
        '''
        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        '''
        tss = np.asarray(hr) (644,1024,3)
        print tss.shape
        '''
        filename = self.hr[index].split("/")[-1]
        #keep the last string, which is after /. For example, a= 'aaa/asds.bmp', a.split("/")[-1] = asds 
        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)

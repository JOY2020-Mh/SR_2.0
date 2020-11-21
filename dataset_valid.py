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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath, inYCbCr=False):
    if inYCbCr:
        img = Image.open(filepath).convert('YCbCr') 
    else:
        img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, GT_dir, LR_dir, is_gray=False, random_scale=False, crop_size=128, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4, save_inImg=False, inYCbCr=True):
        super(TrainDatasetFromFolder, self).__init__()
        self.GT_image_filenames = []
        self.LR_image_filenames = []
        for image_name in sorted(glob.glob(GT_dir+'/*')):
            self.GT_image_filenames.append(image_name)
        for image_name in sorted(glob.glob(LR_dir+'/*')):
            self.LR_image_filenames.append(image_name)


        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor
        self.save_inImg = save_inImg
        self.inYCbCr = inYCbCr
        #print(self.GT_image_filenames)
    def __getitem__(self, index):
        # load image
        # print self.GT_image_filenames[index]
        # print self.LR_image_filenames[index]
        GT_img = load_img(self.GT_image_filenames[index], self.inYCbCr)
        LR_img = load_img(self.LR_image_filenames[index], self.inYCbCr)
	    #downscale LR image to 1/2
        lr_h = LR_img.size[1] // 2
        lr_w = LR_img.size[0] // 2
        #print(self.GT_image_filenames[index])
        name = 'img_HR'+str(index)+'.png'
        #print(name)

        #print(GT_img.size)
        transform = Resize((lr_h, lr_w), interpolation=Image.BILINEAR)
        LR_img = transform(LR_img)
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
        hr_w = GT_img.size[0]
        hr_h = GT_img.size[1]

        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        img_LR = LR_img.crop((rnd_w, rnd_h, rnd_w + lr_crop_w, rnd_h + lr_crop_h))
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale_factor), int(rnd_w * self.scale_factor)
        img_HR = GT_img.crop((rnd_w_HR, rnd_h_HR, rnd_w_HR + hr_crop_w, rnd_h_HR + hr_crop_h))
        #print('img_HR.size', img_HR.size)
        #print('img_LR.size', img_LR.size)

        # if self.save_inImg == True:
        #     imageio.imsave('train_1113_only_cropped1113_test/img_HR'+str(index)+'.png', img_HR)
        #     imageio.imsave('train_1113_only_cropped1113_test/img_LR'+str(index)+'.png', img_LR)
        # #   img_HR.save('train_image/img_HR'+str(index)+'.png')
        #   img_LR.save('train_image/img_LR'+str(index)+'.png')

        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(p=1)
            img_HR = transform(img_HR)
            img_LR = transform(img_LR)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(p=1)
            img_HR = transform(img_HR)
            img_LR = transform(img_LR)
        
        # if self.save_inImg == True:
        #      imageio.imsave('0_cropped_flip_1113_test/img_HR'+str(index)+'.png', img_HR)
        #      imageio.imsave('0_cropped_flip_1113_test/img_LR'+str(index)+'.png', img_LR)
       
        
        
        p2 = random.random()
        if p2 >= 2/3:
            img_HR.transpose(Image.ROTATE_90)
            img_LR.transpose(Image.ROTATE_90)
        elif 2/3 > p2 >= 1/3:
            img_HR.transpose(Image.ROTATE_180)
            img_LR.transpose(Image.ROTATE_180)
        else:
            img_HR = img_HR
            img_HR = img_HR

        # if self.save_inImg == True:
        #     imageio.imsave('0_cropped_flip_rotate_1113_test111/img_HR'+str(index)+'.png', img_HR)
        #     imageio.imsave('0_cropped_flip_rotate_1113_test111/img_LR'+str(index)+'.png', img_LR)
    
        # angle_choice = [180, 90, 0, -90, -180]
        # angle = random.choice(angle_choice)
        # img_HR = img_HR.rotate(angle)
        # img_LR = img_LR.rotate(angle)
        
        

        if self.save_inImg == True:
            imageio.imsave('train_1114/HR/img_HR'+str(index)+'.png', img_HR)
            imageio.imsave('train_1114/LR/img_LR'+str(index)+'.png', img_LR)
        #   img_HR.save('train_image/img_HR'+str(index)+'.png')
        #   img_LR.save('train_image/img_LR'+str(index)+'.png')

        img_HR=ToTensor()(img_HR)
        img_LR=ToTensor()(img_LR)

        return {'img_LR': img_LR, 'img_HR': img_HR}
        ### croped img_HR and img_LR

    def __len__(self):
        return len(self.GT_image_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, test_GT, test_dataset, is_gray=False, random_scale=False, crop_size=128, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4, save_inImg=False, inYCbCr=True):
        super(TestDatasetFromFolder, self).__init__()
        self.GT_image_filenames = []
        self.LR_image_filenames = []
        for image_name in sorted(glob.glob(test_GT+'/*')):
            self.GT_image_filenames.append(image_name)
        for image_name in sorted(glob.glob(test_dataset+'/*')):
            self.LR_image_filenames.append(image_name)


        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor
        self.save_inImg = save_inImg
        self.inYCbCr = inYCbCr

    def __getitem__(self, index):
        # load image
        # print self.GT_image_filenames[index]
        # print self.LR_image_filenames[index]
        GT_img = load_img(self.GT_image_filenames[index], self.inYCbCr)
        LR_img = load_img(self.LR_image_filenames[index], self.inYCbCr)
	    #downscale LR image to 1/2
        lr_h = LR_img.size[1] // 2
        lr_w = LR_img.size[0] // 2

        transform = Resize((lr_h, lr_w), interpolation=Image.BILINEAR)
        LR_img = transform(LR_img)
        # print self.GT_image_filenames[index]+ 'GT,' + self.LR_image_filenames[index] + 'LR'
        # determine valid HR image size with scale factor
        # self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_crop_w = self.crop_size
        hr_crop_h = self.crop_size

        # determine LR crop image size
        lr_crop_w = hr_crop_w // self.scale_factor
        lr_crop_h = hr_crop_h // self.scale_factor

        #ramdom crop image
        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        img_LR = LR_img.crop((rnd_h, rnd_w, rnd_h + lr_crop_h, rnd_w + lr_crop_w))
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale_factor), int(rnd_w * self.scale_factor)
        img_HR = GT_img.crop((rnd_h_HR, rnd_w_HR, rnd_h_HR + hr_crop_h, rnd_w_HR + hr_crop_w))
        # print('img_HR.size', img_HR.size)
        # print('img_LR.size', img_LR.size)
        

        if self.save_inImg == True:
            imageio.imsave('1114_validation_image/img_HR'+str(index)+'.png', img_HR)
            imageio.imsave('1114_validation_image/img_LR'+str(index)+'.png', img_LR)
        #   img_HR.save('train_image/img_HR'+str(index)+'.png')
        #   img_LR.save('train_image/img_LR'+str(index)+'.png')

        img_HR=ToTensor()(img_HR)
        img_LR=ToTensor()(img_LR)

        return {'img_LR': img_LR, 'img_HR': img_HR}
        ### croped img_HR and img_LR

    def __len__(self):
        return len(self.GT_image_filenames)
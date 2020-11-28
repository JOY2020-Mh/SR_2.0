import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils



class SRDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.GT_dir = opt.GT_dir
        self.LR_dir = opt.LR_dir
        #print("self.GT_dir", self.GT_dir)

        self.file_list = []
        #for img_dir in self.LR_dir:
        img_dir = self.LR_dir[0]
        #print(img_dir)

        self.file_list.extend(utils.get_files(img_dir))
        #print(utils.get_files(img_dir))
        self.file_list_hr = []
        #for img_dir_gt in self.GT_dir:
        
        img_dir_gt = self.GT_dir[0]
        self.file_list_hr.extend(utils.get_files(img_dir_gt))
        #print(self.file_list_hr)
        
        self.file_list = sorted(self.file_list)
        self.file_list_hr = sorted(self.file_list_hr)

    def img_aug(self, input_img, gt_img):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            if input_img.shape[0] != input_img.shape[1]:
                rotate = random.choice([0, 2])
            else:
                rotate = random.randint(0, 3)

            if rotate != 0:
                input_img = np.rot90(input_img, rotate)
                gt_img = np.rot90(gt_img, rotate)

            # horizontal flip
            if np.random.random() >= 0.5:
                input_img = cv2.flip(input_img, flipCode = 1)
                gt_img = cv2.flip(gt_img, flipCode = 1)
                
        return input_img, gt_img

    def rgb2ycbcr_norm(self, img):
        ycbcr = img.copy()
        ycbcr[:,:,0] = ((0.257 * img[:,:,0] + 0.504 * img[:,:,1] + 0.098 * img[:,:,2] + 16) - 16)/(235-16)
        ycbcr[:,:,1] = ((-0.148 * img[:,:,0] - 0.291 * img[:,:,1] + 0.439 * img[:,:,2] + 128) - 16)/(240-16)
        ycbcr[:,:,2] = ((0.439 * img[:,:,0] - 0.368 * img[:,:,1] - 0.071 * img[:,:,2] + 128) - 16)/(240-16)
        return ycbcr

    def shave(self, img, border_size):
        img = img[border_size:-border_size, border_size:-border_size,:]
        return img


    def __getitem__(self, index):

        # Define path
        #LR_dir, LR_img_name = os.path.split(self.file_list[index])
        #HR_dir, HR_img_name = os.path.split(self.file_list_hr[index])
        #HR_dir = os.path.join("/".join(LR_dir.split("/")[:-1]), 'HR')
        #if "asf_off" not in LR_dir:
        #    HR_img_name = LR_img_name
        #else:
        #    HR_img_name = "_".join(LR_img_name.split('_')[0:2]) + '_C_Visidon.png
        #print(index)
        #print(HR_img_name)
        #in_path = os.path.join(LR_dir, LR_img_name)
        #out_path = os.path.join(HR_dir, HR_img_name)
        #print(self.file_list[index])

        in_path = self.file_list[index]
        out_path = self.file_list_hr[index]
        # Read images
        # input
        input_img = Image.open(in_path).convert('YCbCr')
        input_img = input_img.crop([0, 0, 125, 125])
        #width, height = input_img.size
        #input_img = input_img.resize((width//2,height//2),Image.BICUBIC)
        input_img = np.array(input_img).astype(np.float32)
        
        # output
        gt_img = Image.open(out_path).convert('YCbCr')
        gt_img = gt_img.crop([0,0,500,500])
        width, height = gt_img.size
        #print(width, height)
        #gt_img = gt_img.resize((width//2,height//2),Image.BICUBIC)
        gt_img = np.array(gt_img).astype(np.float32)

        
        input_img, gt_img = self.img_aug(input_img, gt_img)

        if self.opt.shave:
            if self.opt.scale_factor == 4:
                border_size = self.opt.scale_factor * 2
            else:
                border_size = self.opt.scale_factor * 2 - 2
            
            gt_img = self.shave(gt_img, border_size)

        #rgb2ycbcr and normalize to 0 to 1
        input_ycbcr = self.rgb2ycbcr_norm(input_img)
        output_ycbcr = self.rgb2ycbcr_norm(gt_img)

        input_img = torch.from_numpy(input_ycbcr.transpose(2, 0, 1).astype(np.float32)).contiguous()
        gt_img = torch.from_numpy(output_ycbcr.transpose(2, 0, 1).astype(np.float32)).contiguous()
        
        return input_img, gt_img
    
    def __len__(self):
        return len(self.file_list)


class SRValidDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.test_lr = opt.test_lr
        self.file_list = utils.get_files(self.test_lr)

    def rgb2ycbcr_norm(self, img):
        ycbcr = img.copy()
        ycbcr[:,:,0] = ((0.257 * img[:,:,0] + 0.504 * img[:,:,1] + 0.098 * img[:,:,2] + 16) - 16)/(235-16)
        ycbcr[:,:,1] = ((-0.148 * img[:,:,0] - 0.291 * img[:,:,1] + 0.439 * img[:,:,2] + 128) - 16)/(240-16)
        ycbcr[:,:,2] = ((0.439 * img[:,:,0] - 0.368 * img[:,:,1] - 0.071 * img[:,:,2] + 128) - 16)/(240-16)
        return ycbcr

    def shave(self, img, border_size):
        img = img[border_size:-border_size, border_size:-border_size,:]
        return img


    def __getitem__(self, index):

        # Define path
        in_path = self.file_list[index]
        print(in_path)

        # Read images
        # input
        input_img = Image.open(in_path)
        width, height = input_img.size
        input_img = np.array(input_img).astype(np.float32)

        #rgb2ycbcr and normalize to 0 to 1
        input_ycbcr = self.rgb2ycbcr_norm(input_img)

        input_img = torch.from_numpy(input_ycbcr.transpose(2, 0, 1).astype(np.float32)).contiguous()
        
        return input_img, self.file_list[index]
    
    def __len__(self):
        return len(self.file_list)



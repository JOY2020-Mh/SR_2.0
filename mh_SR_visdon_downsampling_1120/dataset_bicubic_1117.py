### add rotation in the training dataset
import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
import numpy as np
import random

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

    def __getitem__(self, index):
        # load image
        
        #GT_img = load_img(self.GT_image_filenames[index], self.inYCbCr)
        #LR_img = load_img(self.LR_image_filenames[index], self.inYCbCr)
        GT_img = Image.open(self.GT_image_filenames[index])
        hr_box = (0, 0, 3200, 2400)
        GT_img = GT_img.crop(hr_box)
        
        #resize LR image to the needed size
        w = GT_img.size[0]
        h = GT_img.size[1]
        #if index % 3 == 0:
        LR_img = GT_img.resize([w//4, h//4], Image.BICUBIC)
       # elif index % 3 == 1:
       #     LR_img = GT_img.resize([w//4, h//4], Image.BILINEAR)
       # else:
       #     LR_img = GT_img.resize([w//4, h//4], Image.LANCZOS)
 
        #angle_choice = [180, 90, -90, -180]
        #angle = random.choice(angle_choice)
        
        #img_HR = GT_img.rotate(angle)
        #img_LR = LR_img.rotate(angle)
        lr_h = LR_img.size[1]
        lr_w = LR_img.size[0]
        hr_crop_w = self.crop_size
        hr_crop_h = self.crop_size

        lr_crop_size = self.crop_size // self.scale_factor
        lr_crop_h = lr_crop_size
        lr_crop_w = lr_crop_size
        #hr_box = (int(hr_w*0.1), int(hr_h *0.1), int(hr_w * 0.9), int(hr_h * 0.9)) 
        rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
        rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
        #rnd_h = (lr_h - lr_crop_h) // 2
        #rnd_w = (lr_w - lr_crop_w) // 2
        
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

        #if random.random() > 0.5:
         #   transform = transforms.RandomHorizontalFlip(p=1)
          #  img_HR = transform(img_HR)
           # img_LR = transform(img_LR)

       # if random.random() > 0.5:
        #    transform = transforms.RandomVerticalFlip(p=1)
         #   img_HR = transform(img_HR)
          #  img_LR = transform(img_LR)
        
        # if self.save_inImg == True:
        #     imageio.imsave('0_cropped_flip_1113_test/img_HR'+str(index)+'.png', img_HR)
        #     imageio.imsave('0_cropped_flip_1113_test/img_LR'+str(index)+'.png', img_LR)
       
        
        
        #p2 = random.random()
        #if p2 >= 2/3:
         #   img_HR.transpose(Image.ROTATE_90)
          #  img_LR.transpose(Image.ROTATE_90)
        #elif 2/3 > p2 >= 1/3:
         #   img_HR.transpose(Image.ROTATE_180)
          #  img_LR.transpose(Image.ROTATE_180)
       # else:
         #   img_HR = img_HR
        #    img_LR = img_LR

        #if self.save_inImg == True:
         #   imageio.imsave('0_cropped_flip_rotate_1113_test111/img_HR'+str(index)+'.png', img_HR)
          #  imageio.imsave('0_cropped_flip_rotate_1113_test111/img_LR'+str(index)+'.png', img_LR)
    
        # angle_choice = [180, 90, 0, -90, -180]
        # angle = random.choice(angle_choice)
        # img_HR = img_HR.rotate(angle)
        # img_LR = img_LR.rotate(angle)
        
        

       # if self.save_inImg == True:
        #    imageio.imsave('train_1113_5/img_HR'+str(index)+'.png', img_HR)
         #   imageio.imsave('train_1113_5/img_LR'+str(index)+'.png', img_LR)
        #   img_HR.save('train_image/img_HR'+str(index)+'.png')
        #   img_LR.save('train_image/img_LR'+str(index)+'.png')

        
        if self.save_inImg == True:
            img_HR.save('train_image/img_HR'+str(index)+'.png')
            img_LR.save('train_image/img_LR'+str(index)+'.png')
        
        img_HR_yuv = img_HR.convert('YCbCr')
        img_LR_yuv = img_LR.convert('YCbCr')
        
        img_HR_item=ToTensor()(img_HR_yuv)
        img_LR_item=ToTensor()(img_LR_yuv)

        return {'img_LR': img_LR_item, 'img_HR': img_HR_item}
        ### croped img_HR and img_LR

    def __len__(self):
        return len(self.GT_image_filenames)

class TestDatasetFromFolder_LR(data.Dataset):
    def __init__(self, image_dir, inYCbCr=False, scale_factor=4, is_LR=False):
        super(TestDatasetFromFolder_LR, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.inYCbCr = inYCbCr
        self.is_LR = is_LR
        self.scale_factor = scale_factor


    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index], self.inYCbCr)
        #img_y, img_cb, img_cr = img.split()
        #np_img_cb = np.array(img_cb)
        
        #print(np_img_cb[:5, :5])
        #print(img_y)
        #img_y.show()

        # original HR image size
        w = img.size[0]
        h = img.size[1]
        #fovzoom = 0
        #if fovzoom:
        #    img = img.crop((int((h - h//fovzoom) // 2), int((w - w//fovzoom) // 2), int((h + h//fovzoom) // 2), int((w + w//fovzoom) // 2)))
        lr_img = transforms.ToTensor()(img)
        #print(lr_img.size())
        #lr_img_cb = lr_img[1:2, :, :]
        #new_lr_img_cb = lr_img_cb.squeeze(0).numpy()
        #print(new_lr_img_cb[:5, :5])
        #print("new_lr_img_cb[:5, :5] * 255")
        #print(new_lr_img_cb[:5, :5] * 255)

        #print(new_lr_img_cb[:5, :5] * 255 - np_img_cb[:5, :5])
        #print('np.array_equal(new_lr_img_y,np_img_y:)', np.array_equal(new_lr_img_y,np_img_y))
        
        # if self.is_LR == False:
        #     # determine lr_img LR image size, downscale it
        #     hr_crop_w = calculate_valid_crop_size(w, self.scale_factor)
        #     hr_crop_h = calculate_valid_crop_size(h, self.scale_factor)
        #     lr_crop_w = hr_crop_w // self.scale_factor
        #     lr_crop_h = hr_crop_h // self.scale_factor
        #     lr_transform = Compose([Resize((lr_crop_h, lr_crop_w), interpolation=Image.BICUBIC), ToTensor()])
        #     lr_img = lr_transform(img)
        # else:
        #     #keep the same size and return
        #     lr_img = transforms.ToTensor()(img)

        return lr_img


    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder_GT(data.Dataset):
    def __init__(self, image_dir, inYCbCr=False, is_gray=False, scale_factor=4,  is_LR = False):
        super(TestDatasetFromFolder_GT, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.inYCbCr = inYCbCr
        self.is_gray = is_gray
        self.is_LR = is_LR
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index], self.inYCbCr)

        # cal the original HR image size or is HR
        #if self.is_LR == True:
        #    w = img.size[0]*self.scale_factor
        #    h = img.size[1]*self.scale_factor
        #else:
        w = img.size[0]
        h = img.size[1]

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([ToTensor()])
        hr_img = hr_transform(img)

        return hr_img

    def __len__(self):
        return len(self.image_filenames)

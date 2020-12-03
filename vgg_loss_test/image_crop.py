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
from os import listdir
from os.path import join

test_dataset = './Ajun_8x_asf_off4/LR-aligned'

# GT_image_filenames = []
LR_image_filenames = []
# for image_name in sorted(glob.glob(test_GT+'/*')):
#     GT_image_filenames.append(image_name)
LR_image_filenames = [join(test_dataset, x) for x in sorted(listdir(test_dataset))]
# for image_name in sorted(glob.glob(test_dataset+'/*')):
#     print(image_name)
#     LR_image_filenames.append(image_name)

crop_size = 256

for index in range(len(LR_image_filenames)):
    LR_img = Image.open(LR_image_filenames[index])
    #downscale LR image to 1/2
    lr_h = LR_img.size[1]
    lr_w = LR_img.size[0]

    lr_crop_w = crop_size
    lr_crop_h = crop_size

    # center crop
    rnd_h = (lr_h - lr_crop_h)//2
    rnd_w = (lr_w - lr_crop_w)//2
    # #ramdom crop image
    # rnd_h = random.randint(0, max(0, lr_h - lr_crop_h - 1))
    # rnd_w = random.randint(0, max(0, lr_w - lr_crop_w - 1))
    img_LR = LR_img.crop((rnd_w, rnd_h, rnd_w + lr_crop_w, rnd_h + lr_crop_h))
    # rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale_factor), int(rnd_w * self.scale_factor)
    # img_HR = GT_img.crop((rnd_h_HR, rnd_w_HR, rnd_h_HR + hr_crop_h, rnd_w_HR + hr_crop_w))
    #print('img_HR.size', img_HR.size)
    #print('img_LR.size', img_LR.size)

    # imageio.imsave('validation_image/img_HR'+str(index)+'.png', img_HR)
    imageio.imsave('cropped_256_image/img_LR'+str(index)+'.png', img_LR)

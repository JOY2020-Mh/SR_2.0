
from os import listdir
from os.path import join
import random
from PIL import Image
import os
import scipy.io
import scipy.misc
import numpy
import imageio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

training_dataset_dir = 'lj_test_image'
#training_dataset_dir = 'visdon_dataset'
# HR_dir = join(training_dataset_dir, 'HR')

new_HR_dir = join(training_dataset_dir, 'crop_HR')
LR_dir = join(training_dataset_dir, 'LR')
image_filenames = [join(training_dataset_dir, x) for x in sorted(listdir(training_dataset_dir)) if is_image_file(x)]
print(len(image_filenames))

if not os.path.exists(LR_dir):
    os.makedirs(LR_dir, mode=0o777)

if not os.path.exists(new_HR_dir):
    os.makedirs(new_HR_dir, mode=0o777)   

for index in range(len(image_filenames)):
    img = Image.open(image_filenames[index])

    GT_img = Image.open(image_filenames[index])
        
    hr_box = (1000, 800, 3000, 2400 )
    GT_img = GT_img.crop(hr_box)
    w = GT_img.size[0]
    h = GT_img.size[1]


    LR_img = GT_img.resize([w//4, h//4], Image.BICUBIC)
    
    save_HR_name = 'HR_' + str(index) + '.png'
    save_path_HR = join(new_HR_dir, save_HR_name)
    # scipy.misc.imsave(save_path_HR, img)
    imageio.imsave(save_path_HR, GT_img)

    save_LR_name = 'LR_' + str(index) + '.png'
    save_path_LR = join(LR_dir, save_LR_name)
    # scipy.misc.imsave(save_path_LR, LR_img)
    imageio.imsave(save_path_LR, LR_img)
    print(index)    
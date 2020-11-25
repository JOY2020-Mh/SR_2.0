
import argparse, os
import shutil
import glob
import cv2
from PIL import Image
from torchvision.transforms import *
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_img_dir", type=str, default="/home/miaohuan/Documents/out-Irvine_copy", help="input image dir")
parser.add_argument("--output_img_dir", type=str, default="/home/miaohuan/Documents/out-Irvine_copy_rename", help="output image dir")
opt = parser.parse_args()

for image in glob.glob(os.path.join(opt.input_img_dir,'*.png')):
    ##rename new_image
	im_name = os.path.split(image)[1].split('.png')[0]
	new_im_name = im_name.split('_')[3] + '_' + im_name.split('_')[4] + '_' + 'C_Visidon.jpg'
	new_im_name = os.path.split(image)[1]
	
	if not os.path.exists(opt.output_img_dir):
		os.makedirs(opt.output_img_dir, mode = 0o777)

	im.save(os.path.join(opt.output_img_dir,new_im_name))

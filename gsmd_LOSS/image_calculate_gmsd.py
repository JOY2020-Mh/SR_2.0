import torch
from gms_loss import *
from PIL import Image
from torchvision import transforms
from gms_loss import MSGMS_Loss

image_path_1= './lj_test_image/1118_visdon_HR_downsampling_2loss_visstyle/0_SR_x_1105_4.png'
image_path_2 = './lj_test_image/1116_tcl_bright/6_SR_x_1105_4.png'

img_ycbcr_1 = Image.open(image_path_1).convert('YCbCr') 
img_y_1, img_cb_1, img_cr_1 = img_ycbcr_1.split()

img_ycbcr_2 = Image.open(image_path_2).convert('YCbCr') 
img_y_2, img_cb_2, img_cr_2 = img_ycbcr_2.split()

img_y_1 = img_y_1.crop((0,0, 100, 200))
img_y_2 = img_y_2.crop((0,0, 100, 200))

transform = transforms.ToTensor()
Ir = transform(img_y_1).unsqueeze(0)
Ii = transform(img_y_2).unsqueeze(0)

print(Ir.size())

# print(Ir.size())

loss = MSGMS_Loss()
y = loss.forward(Ii, Ir)

print(y)










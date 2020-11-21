'''
This function is for test.
Given the model strcuture;
given the opretrained model;
and the test image file.
You need to change the --test_dataset according to your case
And the result is the upscaled 4 images.
'''

'''
if you use 9 layer###
You need to change Net_new4---Net"
besides, you still need to change m from 1---5
'''

is_train = False;
scale_factor = 4
##save_path = 'Results_x4'
#Model_index = 1  
#pretrained_model = 'vistyle_day_v6_denoise.pkl'
#pretrained_model = 'lr_0.0001_x4_Net_new4_epoch_100.pkl'
#pretrained_model = 'lr_0.0001_x4_Net_epoch_80.pkl'
#pretrained_model = 'lr_0.0001_x4_Net_new4_epoch_430.pkl'

#pretrained_model = 'lr_0.0001_x4_Net_new4_epoch_810.pkl'

#pretrained_model = 'lr_0.0001_x4_Net_epoch_760.pkl'
#pretrained_model = '1103_05_layer_lr_0.0001_x4_Net_new4_epoch_130.pkl'
pretrained_model = '1104_augument_based_1430_epoch110.pkl'
# pretrained_model = '1103_5_ourdata_augument_lr_0.0001_x4_Net_new4_epoch_120.pkl'
#pretrained_model = '1024_trainning_modellr_0.0001_x4_Net_new4_epoch_500.pth'
# pretrained_model = '1103_visdon_lr_0.0001_x4_Net_new4_epoch_200.pkl'
print ('scale_factor = %d' %scale_factor)


import torch
import os, argparse
import numpy as np
from PIL import Image 
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import *
import scipy.io
import numpy
import scipy.misc
import time
import torch._utils
from base_networks import *
from network import *
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
#from TCL_SuperResolution_Model import TCL_SuperResolution
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of S R collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--is_train', type=bool, default=is_train)
    parser.add_argument('--model_name', type=str, default='Net_new4',
                        choices=['Net', 'Net_new4', 'Net_new3'], help='The type of model')
    parser.add_argument('--scale_factor', type=int, default=scale_factor, help='Size of scale factor')
    parser.add_argument('--Model_index', type=int, default= 0)  ### 0 is model used until V1.6. Rest represent the update index.
    parser.add_argument('--loss_func', type=str, default='mse') ##mse, ssim
    parser.add_argument('--GT_dir', type=str, default='./SR_train/HR')
    parser.add_argument('--LR_dir', type=str, default='./SR_train/LR')
    #parser.add_argument('--GT_dir', type=str, default='../data/T1/asf_off/ppt/HR')
    #parser.add_argument('--LR_dir', type=str, default='../data/T1/asf_off/ppt/LR')
    parser.add_argument('--inYCbCr', type=bool, default=False)
    parser.add_argument('--save_inImg', type=bool, default=False)

    parser.add_argument('--is_LR', type=bool, default=True)
    parser.add_argument('--test_dataset', type=str, default= 'test_images/LR',help="the LR of test dataset")
    parser.add_argument('--test_GT', type=str, default='test_images/HR',help="The GT of test dataset")

    #parser.add_argument('--test_GT', type=str, default= './SR_validation/DIV2K_validation_HR') #//'../data/Set5/HR')
    #parser.add_argument('--test_dataset', type=str, default= './SR_validation/DIV_2K_validation_bicubic_4x_LR')#'/home/pancc/eclipse-workspace/TCLSuperRes/data/Btest4/burst_1frame/LR/LRless') # to be logically optimized.......
    parser.add_argument('--model_save_dir', type=str, default='./1024_trainning_model', help='Directory name to save models')
    # parser.add_argument('--img_save_dir', type=str, default='./1104_test_5_layer_1430_epoch', help='Directory name to save validation pictures')
    #parser.add_argument('--pretrained_model', type=str, default='./pretrained_model/'+model_name, help='Directory name to pretrained model')
    parser.add_argument('--img_save_dir', type=str, default= './1104_5_layer_1430')
    parser.add_argument('--pretrained_model', type=str, default= pretrained_model, help='pretrained model')
    parser.add_argument('--crop_size', type=int, default=500, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--num_epochs', type=int, default=2000, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--Word', type=bool, default=False)
    # parser.add_argument('--ckpt_dir', type = str, default = '1024_SR_mh_checkpoint')
    # parser.add_argument("--ckpt_name", type=str, default = '1024_Net_new4')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    assert args.test_dataset, 'Error: test_dataset path is not exist..'
    assert args.pretrained_model, 'Error: pretrained_model path is not exist..'


    return args

""" main """
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # print 'scale factor = ', scale_factor, \
    #     '\ntest_dir =', args.test_dataset,\

    from network import Net_new4 as net
    #from network import Net as net
    model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
    #model = net(num_channels=1, scale_factor=4, d=32, s=5, m=5)
    #model.load_state_dict(torch.load(args.pretrained_model, map_location = torch.device('cpu')))
    #In GPU type
    model.load_state_dict(torch.load(args.pretrained_model))

    # for param_tensor in model.state_dict():
    #     #print(param_tensor)
    #     #print(model.state_dict()[param_tensor].size())
    #     if 'act' in param_tensor:
    #         print(param_tensor)
    #         print(model.state_dict()[param_tensor]) 
    
    # print(model)

    model_name_save = './1104_augument_based_1430_epoch110_3pm.onnx'
    x=torch.randn(1,1,426,570,requires_grad=False).type(torch.float)
    torch_out = model(x)
    torch.onnx.export(model,x,model_name_save,export_params=True)

if __name__ == '__main__':

    main()


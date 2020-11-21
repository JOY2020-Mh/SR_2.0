'''
This function is for test.
Given the model strcuture;
given the opretrained model;
and the test image file.
You need to change the --test_dataset according to your case
And the result is the upscaled 4 images.
'''

is_train = False;
scale_factor = 4
##save_path = 'Results_x4'
Model_index = 1  
#pretrained_model = 'vistyle_day_v6_denoise.pkl'
#pretrained_model = '1104_trained_1480.pkl'
# pretrained_model = 'visdon_1106_lr_0.0001_x4_Net_new4_epoch_450.pkl'
pretrained_model = '1121_Visstyle_Steven_dataset_only_random_crop_multi_GPU_Net_epoch_1300.pkl'
#pretrained_model = '1116_visdon_based_visstyle_lr_0.0001_x4_Net_epoch_250.pkl'


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
import imageio

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
    parser.add_argument('--test_dataset', type=str, default= 'lj_test_image',help="the LR of test dataset")
    #parser.add_argument('--test_dataset', type=str, default= '1109Urban_dataset/LR',help="the LR of test dataset")
    #parser.add_argument('--test_dataset', type=str, default= '1109test_images/LR',help="the LR of test dataset")
    parser.add_argument('--test_GT', type=str, default='/HR',help="The GT of test dataset")

    #parser.add_argument('--test_GT', type=str, default= './SR_validation/DIV2K_validation_HR') #//'../data/Set5/HR')
    #parser.add_argument('--test_dataset', type=str, default= './SR_validation/DIV_2K_validation_bicubic_4x_LR')#'/home/pancc/eclipse-workspace/TCLSuperRes/data/Btest4/burst_1frame/LR/LRless') # to be logically optimized.......
    parser.add_argument('--model_save_dir', type=str, default='./1106_train_epoch_450_based1430', help='Directory name to save models')
   #parser.add_argument('--img_save_dir', type=str, default='./a_lj_test_1109_visdon_1340_1430', help='Directory name to save validation pictures')
    #parser.add_argument('--pretrained_model', type=str, default='./pretrained_model/'+model_name, help='Directory name to pretrained model')
    #parser.add_argument('--img_save_dir', type=str, default='./1109test_images/1109_DIV_train', help='Directory name to save validation pictures')
    parser.add_argument('--img_save_dir', type=str, default='lj_test_image/1121_test_1300', help='Directory name to save validation pictures')
    parser.add_argument('--pretrained_model', type=str, default= pretrained_model, help='pretrained model')
    parser.add_argument('--crop_size', type=int, default=500, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--num_epochs', type=int, default=2000, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--Word', type=bool, default=False)
    parser.add_argument('--ckpt_dir', type = str, default = '1024_SR_mh_checkpoint')
    parser.add_argument("--ckpt_name", type=str, default = '1024_Net_new4')
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


    ####method 1 Using  model = torch.nn.DataParallel(model)
    # from network import Net as net
    # model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
    # model = torch.nn.DataParallel(model)
    # state_dict = torch.load(pretrained_model, map_location = torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # print(model)
    # model_name_save = '1120_Visstyle_Steven_dataset_only_random_crop_multi_GPU_Net_based_tcl_epoch_920.pth'
    # torch.save(model, model_name_save)

    ####method 2 rename the key 
    from network import Net as net
    model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
    # original saved file with DataParallel
    state_dict = torch.load(pretrained_model, map_location = torch.device('cpu'))
    #print(state_dict)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict)   
    print(model)


    image_dir = args.test_dataset
    image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
    file_num = len(image_filenames)
    for idx in range(file_num):
        img_ycbcr = Image.open(image_filenames[idx]).convert('YCbCr') 
        img_y, img_cb, img_cr = img_ycbcr.split()
        #print(img_y.size, img_cb.size, img_cr.size)
    
        input_x_t = torch.from_numpy(numpy.zeros((1, 1, np.array(img_y).shape[0], np.array(img_y).shape[1]), dtype='f')) 
        #temp = torch.from_numpy(np.array(img_y))

        input_x_t[0, 0, :, :] = torch.from_numpy(np.array(img_y)/255)
        with torch.no_grad():
            recon_y = model(input_x_t)
        temp_y = recon_y[0,0,:,:].numpy() * 255
        out_y = Image.fromarray(np.uint8(temp_y.clip(0,255)), mode="L")

        # out_y.show()
        ### use ToPILImage()
        #temp_y = torch.tensor(recon_y.squeeze(0).squeeze(0),dtype = torch.uint8)
        #out_y = transforms.ToPILImage()(temp_y)
        out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
        #out_cb.show()
        out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
        #out_cr.show()
        result = Image.merge('YCbCr',[out_y, out_cb, out_cr]).convert('RGB')
        #result.show()
        image_dir = args.img_save_dir

        if not os.path.exists(image_dir):
            os.mkdir(image_dir, mode=0o777)
        
        save_path = join(image_dir, str(idx)+'_SR_x_1105'+'_' + str(4)+'.png')
        imageio.imsave(save_path, result)
        #scipy.misc.imsave(save_path, result)
        idx += 1

        #print(idx)

    # save entire model
    

    model_name_save = '1024_trainning_model_lr_0.0001_x4_Net_new4_epoch_500.pth'
    torch.save(model, model_name_save)

    # from network import Net as net
    # model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
    # model.load_state_dict(torch.load(args.pretrained_model))

    # model_name_save = './1104_trained_1480.onnx'
    # x=torch.randn(1,1,125,125,requires_grad=False).type(torch.float)
    # torch_out = model(x)
    # torch.onnx.export(model,x,model_name_save,export_params=True)

if __name__ == '__main__':

    main()


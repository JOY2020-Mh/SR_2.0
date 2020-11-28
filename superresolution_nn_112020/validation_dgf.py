import argparse
import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

import dataset
import utils

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_root', type = str, default = "../ECCV_UDC/valid_Poled", help = 'the testing folder')
    parser.add_argument('--saving_root', type = str, default = "./test_dgf", help = 'the testing folder')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = './model/epoch295_batchsize4', help = 'test model name')
    parser.add_argument('--load_DGF_name', type = str, default = './DGF_model/epoch96_batchsize1', help = 'test model name')
    parser.add_argument('--load_combine_DGF_name', type = str, default = './DGF_model/epoch3_batchsize1', help = 'test model name')

    parser.add_argument('--load_pre_train', type = bool, default = True, help = 'load pre-train weight or not')
    parser.add_argument('--load_DGF_pre_train', type = bool, default = True, help = 'load pre-train weight or not')
    parser.add_argument('--load_combine_DGF_pre_train', type = bool, default = True, help = 'load pre-train weight or not')
    

    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')

    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input number of channel')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output number of channel')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')


    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.DGFUDCValidDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = utils.create_generator(opt)
    dgf_model = utils.create_DGF_generator(opt)
    

    model.cuda()
    dgf_model.cuda()
    
    model.eval()
    dgf_model.eval()

    with torch.no_grad():
    
        for batch_idx, (img_input, input_hr) in enumerate(dataloader):

                img_input = img_input.cuda()
                input_hr = input_hr.cuda()
                
                # Generator output
                output_lr = model(img_input)
                img_input = img_input * 0.5 + 0.5
                output_lr = output_lr * 0.5 + 0.5

                output_hr = dgf_model(img_input, output_lr, input_hr)

                _,_,height,width = output_hr.shape

                output_hr = output_hr.detach().cpu().numpy().reshape(3, height, width).transpose(1, 2, 0)
                output_hr = output_hr * 255.0
                output_hr = output_hr.astype(np.uint8)

                
                r, g, b = cv2.split(output_hr)
                show_img = cv2.merge([b, g, r])
                valid_img_save_path = os.path.join(opt.saving_root, 'result_%d_epoch_%d.png' % (batch_idx,96))
                print(valid_img_save_path)
                cv2.imwrite(valid_img_save_path, show_img)


            

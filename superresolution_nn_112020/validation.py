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
    parser.add_argument('--valid_root', type = str, default = "../TCL_UDC_dataset/OV24B_arcsoft", help = 'the testing folder')
    parser.add_argument('--saving_root', type = str, default = "./OV24B_arcsoft", help = 'the testing folder')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = './model/epoch21_batchsize1', help = 'test model name')
    parser.add_argument('--load_pre_train', type = bool, default = True, help = 'load pre-train weight or not')

    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input number of channel')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output number of channel')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')

    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    SRValidDataset
    testset = dataset.SRValidDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = utils.create_generator(opt)
    model.cuda()
    model.eval()

    with torch.no_grad():
    
        for batch_idx, (img_input) in enumerate(dataloader):

                img_input = img_input.cuda()
                
                # Generator output
                fake1 = model(img_input)
                # fake1 = img_input.cuda()

                _,_,height,width = fake1.shape

                fake_1 = fake1.detach().cpu().numpy().reshape(3, height, width).transpose(1, 2, 0)
                fake_1 = (fake_1 * 0.5 + 0.5) * 255.0
                fake_1 = fake_1.astype(np.uint8)

                
                r, g, b = cv2.split(fake_1)
                show_img = cv2.merge([b, g, r])
                valid_img_save_path = os.path.join(opt.saving_root, 'result_%d_epoch_%d.png' % (batch_idx,21))
                print(valid_img_save_path)
                cv2.imwrite(valid_img_save_path, show_img)


                fake_1 = img_input.detach().cpu().numpy().reshape(3, height, width).transpose(1, 2, 0)
                fake_1 = (fake_1 * 0.5 + 0.5) * 255.0
                fake_1 = fake_1.astype(np.uint8)

                
                # r, g, b = cv2.split(fake_1)
                # show_img = cv2.merge([b, g, r])
                # valid_img_save_path = os.path.join(opt.saving_root, 'result_%d_epoch_%d_input.png' % (batch_idx,21))
                # print(valid_img_save_path)
                # cv2.imwrite(valid_img_save_path, show_img)


                # fake_1 = gt_img.numpy().reshape(3, height, width).transpose(1, 2, 0)
                # fake_1 = (fake_1 * 0.5 + 0.5) * 255.0
                # fake_1 = fake_1.astype(np.uint8)

                
                # r, g, b = cv2.split(fake_1)
                # show_img = cv2.merge([b, g, r])
                # valid_img_save_path = os.path.join(opt.saving_root, 'result_%d_epoch_%d_gt.png' % (batch_idx,21))
                # print(valid_img_save_path)
                # cv2.imwrite(valid_img_save_path, show_img)


            

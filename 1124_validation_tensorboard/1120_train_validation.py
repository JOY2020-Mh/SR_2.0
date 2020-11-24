
#is_valid = True
is_train = True
scale_factor = 4
Model_index = 2   #0 for no loading pretrained model, 1 for loading pretrained model
#model_name = '4x/visidon_style/vistyle_day_v2.pkl'
model_name = 'Net'
print ('scale_factor = %d' %scale_factor)
pretrained_model = 'tcl_x4_bright.pkl'
from TCL_SuperResolution_Model_validation_1120 import TCL_SuperResolution

import torch
import os, argparse


"""parsing and configuration"""

def parse_args():
    desc = "PyTorch implementation of S R collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--is_train', type=bool, default=is_train)
    #parser.add_argument('--is_valid', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='Net',
                        choices=['Net', 'Net_new4', 'Net_new3'], help='The type of model')
    parser.add_argument('--scale_factor', type=int, default=scale_factor, help='Size of scale factor')
    parser.add_argument('--Model_index', type=int, default= 2)  ### 0 is model used until V1.6. Rest represent the update index.
    parser.add_argument('--loss_func', type=str, default='ssim') ##mse, ssim
    parser.add_argument('--GT_dir', type=str, default='/home/miaohuan/Documents/visdon_pair_training_dataset/HR')
    parser.add_argument('--LR_dir', type=str, default='/home/miaohuan/Documents/visdon_pair_training_dataset/merge_LR')
    #parser.add_argument('--GT_dir', type=str, default='../data/T1/asf_off/ppt/HR')
    #parser.add_argument('--LR_dir', type=str, default='../data/T1/asf_off/ppt/LR')
    parser.add_argument('--inYCbCr', type=bool, default=False)
    parser.add_argument('--save_inImg', type=bool, default=False)

    parser.add_argument('--is_LR', type=bool, default=True)
    parser.add_argument('--test_GT', type=str, default= '/home/miaohuan/Documents/paired_visdon/T1-visdon-pair/png_T1_cropped') #//'../data/Set5/HR')
    parser.add_argument('--test_dataset', type=str, default= '/home/miaohuan/Documents/paired_visdon/T1-visdon-pair/T1-merge')
    #parser.add_argument('--test_GT', type=str, default= '/home/miaohuan/Documents/imdownsampling_validation_dataset/T1_2_HR_for_downsampling') 
    #parser.add_argument('--test_dataset', type=str, default= '/home/miaohuan/Documents/imdownsampling_validation_dataset/LR_bicubic')
    parser.add_argument('--model_save_dir', type=str, default='./1120_trainning_model', help='Directory name to save models')
    parser.add_argument('--img_save_dir', type=str, default='./experiments/1120', help='Directory name to save validation pictures')
    parser.add_argument('--pretrained_model', type=str, default= pretrained_model, help='Directory name to pretrained model')

    parser.add_argument('--crop_size', type=int, default=500, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--num_epochs', type=int, default=20000, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=1, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--Word', type=bool, default=False)
    parser.add_argument('--decay', type = int, default = 2000)
    # parser.add_argument('--ckpt_dir', type = str, default = '1024_SR_mh_checkpoint')
    # parser.add_argument("--ckpt_name", type=str, default = '1024_Net_new4')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --model_save_dir
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.img_save_dir):
        os.makedirs(args.img_save_dir)


    assert args.GT_dir, 'Error: GT_dir path is not exist.'
    assert args.LR_dir, 'Error: LR_dir path is not exist..'
    assert args.test_dataset, 'Error: test_dataset path is not exist..'
    assert args.pretrained_model, 'Error: pretrained_model path is not exist..'
    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

""" main """
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    print('scale factor = ', scale_factor,\
        'crop_size = ', args.crop_size,\
        '\nlr = ', args.lr, \
        '\nGT_dir =', args.GT_dir, \
        '\nLR_dir =', args.LR_dir, \
        '\nimg_save_dir =', args.img_save_dir, \
        '\nmodel_save_dir =', args.model_save_dir,\
        '\nModel_index =', args.Model_index,\

    )


    # model
    model = TCL_SuperResolution(args)
    model.train()
    



if __name__ == '__main__':

    main()


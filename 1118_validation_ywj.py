
is_valid = True
is_train = False
scale_factor = 4
Model_index = 2   #0 for no loading pretrained model, 1 for loading pretrained model
#model_name = '4x/visidon_style/vistyle_day_v2.pkl'
model_name = 'Net'
print ('scale_factor = %d' %scale_factor)
#pretrained_model = '1118_visdon_HR_downsampling_ssim_Net_epoch_90.pkl'
from TCL_SuperResolution_Model_Visdon_validation_1117 import TCL_SuperResolution

import torch
import os, argparse


"""parsing and configuration"""

class Args():
    def __init__(self, scale_factor, pretrained_model):
        self.is_train = is_train
        self.is_valid = is_valid
        self.is_LR = True
        self.model_name = 'Net'
        self.scale_factor=scale_factor
        self.Model_index= 2
        self.loss_func='ssim'
        self.GT_dir='/home/miaohuan/Documents/visdon_pair_training_dataset/HR'
        self.LR_dir='/home/miaohuan/Documents/visdon_pair_training_dataset/merge_LR'
        self.inYCbCr = False
        self.save_inImg = False
        self.test_GT = '/home/miaohuan/Documents/paired_visdon/T1-visdon-pair/png_T1_cropped'
        self.test_dataset = '/home/miaohuan/Documents/paired_visdon/T1-visdon-pair/T1-merge'
        self.model_save_dir='./1118_trainning_model'
        self.img_save_dir='./experiments/1028'
        self.pretrained_model = pretrained_model

        self.crop_size = 500
        self.num_threads=0
        self.num_channels=1
        self.num_epochs=20000
        self.save_epochs=1
        self.batch_size=8
        self.test_batch_size=1
        self.lr=0.0001
        self.gpu_mode=False
        self.Word=False
        self.decay = 2000

# """checking arguments"""
# def check_args(args):
#     # --model_save_dir
#     if not os.path.exists(args.model_save_dir):
#         os.makedirs(args.model_save_dir)
#     if not os.path.exists(args.img_save_dir):
#         os.makedirs(args.img_save_dir)


#     assert args.GT_dir, 'Error: GT_dir path is not exist.'
#     assert args.LR_dir, 'Error: LR_dir path is not exist..'
#     assert args.test_dataset, 'Error: test_dataset path is not exist..'
#     assert args.pretrained_model, 'Error: pretrained_model path is not exist..'
#     # --epoch
#     try:
#         assert args.num_epochs >= 1
#     except:
#         print('number of epochs must be larger than or equal to one')

#     # --batch_size
#     try:
#         assert args.batch_size >= 1
#     except:
#         print('batch size must be larger than or equal to one')

#     return args

""" main """
def main():
    # parse arguments
    pkl_file = './previous_file/1116_pkl_file/'
    for filename in os.listdir(pkl_file):
        if os.path.splitext(filename)[1] == '.pkl':
            print(filename)
            pretrained_model = pkl_file + filename

            args = Args(scale_factor, pretrained_model)
     
            model = TCL_SuperResolution(args)
            model.validation()
    



if __name__ == '__main__':

    main()


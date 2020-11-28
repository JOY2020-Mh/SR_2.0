import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--load_pre_train', type = bool, default = True, help = 'load pre-train weight or not')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--valid_by_epoch', type = int, default = 1, help = 'interval between model validation (by epochs)')
    parser.add_argument('--save_name_mode', type = bool, default = True, help = 'True for concise name, and False for exhaustive name')
    parser.add_argument('--load_name', type = str, default = 'pretrain_model/tcl_x4_bright.pkl', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--model_offset', type = int, default = 0, help = 'model offset num')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 500, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate for G') 
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 100, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 200000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--num_channels', type = int, default = 1, help = 'input number of channel')
    parser.add_argument('--scale_factor', type = int, default = 4, help = 'scale factor')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
   
    # Dataset parameters
    parser.add_argument('--task', type = str, default = 'SR', help = 'the specific task of the system')
    parser.add_argument('--angle_aug', type = bool, default = True, help = 'data augmentation')
    parser.add_argument('--shave', type = int, default = 1, help = 'crop GT image base on network output resolution, 0 is False, 1 is True')
    parser.add_argument('--GT_dir', nargs="+", type=str, default=['../data/asf_off1/HR', '../data/asf_off2/HR', '../data/asf_off3/HR', '../data/asf_off4/HR', '../data/T1-1/HR', '../data/T1-2/HR', '../data/Seattle-1/HR', '../data/Seattle-2/HR'], help = 'GT directory')
    parser.add_argument('--LR_dir', nargs="+", type=str, default=['../data/asf_off1/LR', '../data/asf_off2/LR', '../data/asf_off3/LR', '../data/asf_off4/LR', '../data/T1-1/LR', '../data/T1-2/LR', '../data/Seattle-1/LR', '../data/Seattle-2/LR'], help = 'LR directory')
    parser.add_argument('--test_lr', type=str, default= '../data/test_asf_off_tct/noasfmerge1', help = 'Test LR directory')
    parser.add_argument('--valid_folder', type=str, default= './valid', help = 'Test output directory')

    # loss parameters
    parser.add_argument('--loss', nargs="+", type=str, default=['ssim','vgg', 'tv'], help = 'loss function: l1, ssim, gradient loss, vgg loss, color loss, tv loss')
    parser.add_argument('--lambda_l1', type=float, default=1, help = 'weighting of l1 loss')
    parser.add_argument('--lambda_ssim', type=float, default=1, help = 'weighting of ssim loss')
    parser.add_argument('--lambda_grad', type=float, default=0.1, help = 'weighting of gradient loss')
    parser.add_argument('--lambda_color', type=float, default=0.1, help = 'weighting of color loss')
    parser.add_argument('--lambda_tv', type=float, default=0.1, help = 'weighting of total variation loss')
    parser.add_argument('--vgg_indices', nargs="+", type=int, default=[21], help = 'vgg loss layer e.g. 2 7 12 21 30')
    parser.add_argument('--lambda_vgg', nargs="+", type=float, default=[0.1], help = 'weighting of total variation loss e.g. 0.05 0.05 0.05 0.1 0.5')

    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    

    print('training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.8f] [Saving mode: %s]'
        % (opt.epochs, opt.batch_size, opt.lr_g, opt.save_mode))

    trainer.train(opt)

        


    

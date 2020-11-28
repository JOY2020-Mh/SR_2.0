import numpy as np
import os
from scipy.io.matlab.mio import savemat, loadmat
import cv2
from torchvision import transforms
import torch
import torch.nn as nn
import network
import argparse
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import time
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def valid_model(testloader, net, epoch, opt):
    net.eval()
    with torch.no_grad():
        for i, (true_input, img_name) in enumerate(testloader):
            if opt.num_channels == 1:
                fake_target = net(true_input[:,0:1,:,:].cuda())

                h = fake_target.shape[2]
                w = fake_target.shape[3]
                y = fake_target[0].clamp(0, 1).detach().cpu().numpy().reshape(1, h, w)[0]
                y = y * (235-16) + 16

                true_input_yuv = F.interpolate(true_input, size=[h, w], mode="nearest")
                true_input_yuv = true_input_yuv[0].detach().cpu().numpy().transpose((1,2,0))
                true_input_yuv[:,:,0] = y[:,:]
                true_input_yuv[:,:,1] = true_input_yuv[:,:,1] * (240-16) + 16
                true_input_yuv[:,:,2] = true_input_yuv[:,:,2] * (240-16) + 16

                y_hr = true_input_yuv.copy()
                y_hr[:,:,2] = 1.164 * (y-16) + 1.596 * (true_input_yuv[:,:,2]-128)
                y_hr[:,:,1] = 1.164 * (y-16) -0.812 * (true_input_yuv[:,:,2]-128) - 0.392 * (true_input_yuv[:,:,1]-128)
                y_hr[:,:,0] = 1.164 * (y-16) + 2.017 * (true_input_yuv[:,:,1]-128)

                out_img_name = os.path.split(img_name[0])[1].split('.jpg')[0] + '_epoch_' + str(epoch) + '.png'
                if not os.path.exists(opt.valid_folder):
                    os.mkdir(opt.valid_folder)

                output_img = os.path.join(opt.valid_folder, out_img_name)
                cv2.imwrite(output_img, y_hr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_lr', type=str, default= '../data/test_asf_off_tct/noasfmerge1', help = 'Test LR directory')
    parser.add_argument('--valid_folder', type=str, default= './test', help = 'Test output directory')
    parser.add_argument('--load_pre_train', type = bool, default = True, help = 'load pre-train weight or not')
    parser.add_argument('--load_name', type = str, default = 'pretrain_model/tcl_x4_bright.pkl', help = 'load the pre-trained model with certain epoch')
    opt = parser.parse_args()
    print(opt)

    testset = dataset.SRValidDataset(opt)
    testloader = DataLoader(testset, batch_size = 1, pin_memory = True)
    

    work_dir = opt.saving_mat_dir
    generator = torch.load(opt.load_name)
    generator = generator.cuda()
    generator.eval()

    # load noisy images
    udc_fn = opt.valid_file
    udc_key = 'test_display'
    udc_mat = loadmat(os.path.join(udc_fn))[udc_key]

    run_time = 0
    # restoration
    n_im, h, w, c = udc_mat.shape
    results = udc_mat.copy()
    for i in range(n_im):
        udc = np.reshape(udc_mat[i, :, :, :], (h, w, c))
        restored, run_time = restoration(opt,generator,udc, i, run_time)
        results[i, :, :, :] = restored
    print(run_time)

    # create results directory
    res_dir = 'res_dir'
    os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

    # save images in a .mat file with dictionary key "results"
    res_fn = os.path.join(work_dir, res_dir, 'results.mat')
    res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
    savemat(res_fn, {res_key: results})

    # submission indormation
    # TODO: update the values below; the evaluation code will parse them
    runtime = run_time/30  # seconds / image
    cpu_or_gpu = 0  # 0: GPU, 1: CPU
    method = 1  # 0: traditional methods, 1: deep learning method
    other = '(optional) any additional description or information'

    # prepare and save readme file
    readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
    with open(readme_fn, 'w') as readme_file:
        readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
        readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
        readme_file.write('Method: %s\n' % str(method))
        readme_file.write('Other description: %s\n' % str(other))

    # compress results directory
    res_zip_fn = 'results_dir'
    shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))

import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import cv2
import os

import dataset
import utils
from loss import *

def train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    vgg_indices = opt.vgg_indices
    vggloss_weights = opt.lambda_vgg

    # Loss functions
    if 'l1' in opt.loss:
        criterion_L1 = torch.nn.L1Loss().cuda()
    
    if 'ssim' in opt.loss:
        ssimLoss = SSIM().cuda()

    if 'tv' in opt.loss:
        totalvar_loss = tv_loss().cuda()

    if 'color' in opt.loss:
        csColorLoss = color_loss().cuda()

    if 'grad' in opt.loss:
        gradLoss = GradientLoss().cuda()

    if 'vgg' in opt.loss:
        vggLoss = VGGLoss(opt).cuda().eval()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if epoch % opt.save_by_epoch == 0:
                torch.save(net.module, './model/epoch%d_batchsize%d.pth' % (epoch + opt.model_offset, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.save_by_epoch == 0:
                torch.save(net, './model/epoch%d_batchsize%d.pth' % (epoch + opt.model_offset, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))

    # Valid the model
    def valid_model(testloader, net, epoch, opt):
        if epoch % opt.valid_by_epoch == 0:
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
                        
                        print(out_img_name)
                        output_img = os.path.join(opt.valid_folder, out_img_name)
                        cv2.imwrite(output_img, y_hr)


    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.SRDataset(opt)
    print(len(trainset))
    testset = dataset.SRValidDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    testloader = DataLoader(testset, batch_size = 1, pin_memory = True)
    #test_dataloader = DataLoader(testset, batch_size = 1, pin_memory = True)
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):

        valid_model(testloader, generator,(epoch + 1), opt)

        avg_l1_loss = 0
        avg_ssim_loss = 0
        avg_cs_ColorLoss = 0
        avg_grad_loss = 0
        avg_ssim_loss_lf = 0
        avg_vgg_loss = 0
        avg_tv_loss = 0

        generator.train()
        for i, (true_input, true_target) in enumerate(dataloader):

            if opt.num_channels == 1:
                true_input = true_input[:,0:1,:,:]
                true_target = true_target[:,0:1,:,:]
            
            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input)

            # overall loss
            loss = 0

            disp_Pixellevel_L1_Loss = 0
            disp_tot_v_loss = 0
            disp_cs_ColorLoss = 0
            disp_grad_loss = 0
            disp_vgg_loss = 0
            disp_ssim_loss = 0


            # L1 Loss
            if 'l1' in opt.loss:
                Pixellevel_L1_Loss = opt.lambda_l1 * criterion_L1(fake_target, true_target)
                loss += Pixellevel_L1_Loss
                avg_l1_loss += Pixellevel_L1_Loss.item()
                disp_Pixellevel_L1_Loss = Pixellevel_L1_Loss.item()
            # tv Loss
            if 'tv' in opt.loss:
                tot_v_loss = opt.lambda_tv * totalvar_loss(fake_target)
                #print(tot_v_loss.type)
                loss += tot_v_loss
                avg_tv_loss += tot_v_loss.item()
                disp_tot_v_loss = tot_v_loss.item()


            # color loss
            if 'color' in opt.loss:
                cs_ColorLoss = opt.lambda_color * csColorLoss(fake_target, true_target)
                loss += cs_ColorLoss
                avg_cs_ColorLoss += cs_ColorLoss.item()
                disp_cs_ColorLoss = cs_ColorLoss.item()

            # gradient loss
            if 'grad' in opt.loss:
                grad_loss = opt.lambda_grad * gradLoss(fake_target, true_target)
                loss += grad_loss
                avg_grad_loss += grad_loss.item()
                disp_grad_loss = grad_loss.item()

            # vgg loss
            if 'vgg' in opt.loss:

                if opt.num_channels == 1:
                    vgg_loss = vggLoss(fake_target.clamp(0, 1).expand(-1,3,-1,-1), true_target.clamp(0, 1).expand(-1,3,-1,-1))
                else:
                    vgg_loss = vggLoss(fake_target.clamp(0, 1), true_target.clamp(0, 1))
                loss += vgg_loss
                avg_vgg_loss += vgg_loss.item()
                disp_vgg_loss = vgg_loss.item()
           
            # ssim loss
            if 'ssim' in opt.loss:
                ssim_loss = 1 - ssimLoss(fake_target, true_target)
                loss += ssim_loss
                print(loss.data)
                avg_ssim_loss += ssim_loss.item()
                disp_ssim_loss = ssim_loss.item()
             
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [color loss: %.4f] [ssim Loss: %.4f] [grad Loss: %.4f] [VGG Loss: %.4f] [TV Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), disp_Pixellevel_L1_Loss, disp_cs_ColorLoss, disp_ssim_loss, disp_grad_loss, disp_vgg_loss, disp_tot_v_loss, time_left))

        # Save model at certain epochs or iterations
        save_model(generator, (epoch + 1), opt)

        # Learning rate decrease at certain epochs
        adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

        avg_l1_loss = avg_l1_loss / (i + 1)
        avg_ssim_loss = avg_ssim_loss / (i + 1)
        avg_cs_ColorLoss = avg_cs_ColorLoss / (i + 1)
        avg_grad_loss = avg_grad_loss / (i + 1)
        avg_vgg_loss = avg_vgg_loss / (i + 1)
        avg_tv_loss = avg_tv_loss / (i + 1)

        f = open("log.txt", "a")
        f.write('epoch: ' + str(epoch) + ' avg l1 =' + str(avg_l1_loss) + ' avg color loss =' + str(avg_cs_ColorLoss) + ' avg ssim = ' + str(avg_ssim_loss) + ' avg grad loss = ' + str(avg_grad_loss) + ' avg vgg loss = ' + str(avg_vgg_loss) + 'avg tv loss = ' + str(avg_tv_loss) +  '\n')
        f.close()

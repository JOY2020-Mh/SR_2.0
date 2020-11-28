import os
import skimage
from PIL import Image
import numpy as np
import torch

import sys
sys.path
sys.path.append('/home/steven/steven/UDC_Model8_concat')

from PerceptualSimilarity.models import PerceptualLoss

lpips_criterion = PerceptualLoss(
    model="net-lin", net="alex", use_gpu=True, gpu_ids=[0]
)


GT_path = '../ECCV_UDC/valid_Poled_GT'
valid_img_path = '../UDC_Model11_concat/test_2'

valid_img_path_2 = '../deep-atrous-guided-filter-master/outputs/ours-poled/val_latest_epoch_959'
#valid_img_path_2 = '../UDC_Model8_concat/test_1'
valid_img_path_3 = '../ECCV_UDC_Self_Attn_34_no_local_global/test_full'

avg_psnr = 0
avg_ssim = 0
avg_lpip = 0


avg_psnr2 = 0
avg_ssim2 = 0
avg_lpip2 = 0


avg_psnr3 = 0
avg_ssim3 = 0
avg_lpip3 = 0
for i in range(30):
	gt_img = os.path.join(GT_path, str(i) + '.png')
	valid_img = os.path.join(valid_img_path,'result_' + str(i) + '_epoch_563.png')
	# valid_img2 = os.path.join(valid_img_path_2,'result_' + str(i+1) + '_epoch_1400.png')
	valid_img2 = os.path.join(valid_img_path_2, str(i+1) + '.png')
	valid_img3 = os.path.join(valid_img_path_3,'result_' + str(i+1) + '_epoch_245.png')

	print(gt_img)
	print(valid_img)

	gt_img = Image.open(gt_img)
	valid_img = Image.open(valid_img)
	valid_img2 = Image.open(valid_img2)
	valid_img3 = Image.open(valid_img3)

	gt_img = np.array(gt_img)
	valid_img = np.array(valid_img)
	valid_img2 = np.array(valid_img2)
	valid_img3 = np.array(valid_img3)

	psnr = skimage.measure.compare_psnr(gt_img, valid_img)
	ssim = skimage.measure.compare_ssim(gt_img, valid_img, gaussian_weights=True, use_sample_covariance=False,  multichannel=True)

	psnr2 = skimage.measure.compare_psnr(gt_img, valid_img2)
	ssim2 = skimage.measure.compare_ssim(gt_img, valid_img2, gaussian_weights=True, use_sample_covariance=False,  multichannel=True)

	psnr3 = skimage.measure.compare_psnr(gt_img, valid_img3)
	ssim3 = skimage.measure.compare_ssim(gt_img, valid_img3, gaussian_weights=True, use_sample_covariance=False,  multichannel=True)
	
	gt_img = gt_img.astype(np.float)/255.0
	valid_img = valid_img.astype(np.float)/255.0
	valid_img2 = valid_img2.astype(np.float)/255.0
	valid_img3 = valid_img3.astype(np.float)/255.0

	gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).contiguous()
	valid_img = torch.from_numpy(valid_img.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).contiguous()
	valid_img2 = torch.from_numpy(valid_img2.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).contiguous()
	valid_img3 = torch.from_numpy(valid_img3.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).contiguous()
	
	lpip = lpips_criterion(gt_img.to("cuda"),valid_img.to("cuda"))
	lpip = lpip.detach().cpu().numpy()[0][0][0][0]

	lpip2 = lpips_criterion(gt_img.to("cuda"),valid_img2.to("cuda"))
	lpip2 = lpip2.detach().cpu().numpy()[0][0][0][0]

	lpip3 = lpips_criterion(gt_img.to("cuda"),valid_img3.to("cuda"))
	lpip3 = lpip3.detach().cpu().numpy()[0][0][0][0]

	avg_psnr += psnr
	avg_ssim += ssim
	avg_lpip += lpip

	avg_psnr2 += psnr2
	avg_ssim2 += ssim2
	avg_lpip2 += lpip2

	avg_psnr3 += psnr3
	avg_ssim3 += ssim3
	avg_lpip3 += lpip3

	# print(psnr,psnr3,ssim,ssim3)
	print('psnr:',psnr,psnr2,psnr3)
	print('ssim:',ssim,ssim2,ssim3)
	print('lpip:',lpip,lpip2,lpip3)


avg_psnr = avg_psnr / 30
avg_ssim = avg_ssim / 30
avg_lpip = avg_lpip / 30

avg_psnr2 = avg_psnr2 / 30
avg_ssim2 = avg_ssim2 / 30
avg_lpip2 = avg_lpip2 / 30

avg_psnr3 = avg_psnr3 / 30
avg_ssim3 = avg_ssim3 / 30
avg_lpip3 = avg_lpip3 / 30
print('avg 1:', avg_psnr, avg_ssim, avg_lpip)
print('avg 2:', avg_psnr2, avg_ssim2, avg_lpip2)
print('avg 3:', avg_psnr3, avg_ssim3, avg_lpip3)

# GT_path = '../ECCV_UDC/valid_Poled_GT'
# valid_img_path = '../deep-atrous-guided-filter-master/outputs/ours-poled/val_latest_epoch_959'

# avg_psnr = 0
# avg_ssim = 0
# for i in range(1,31):
# 	gt_img = os.path.join(GT_path, str(i-1) + '.png')
# 	valid_img = os.path.join(valid_img_path, str(i) + '.png')
# 	print(gt_img)
# 	print(valid_img)

# 	gt_img = Image.open(gt_img)
# 	valid_img = Image.open(valid_img)

# 	gt_img = np.array(gt_img)
# 	valid_img = np.array(valid_img)

# 	psnr = skimage.measure.compare_psnr(gt_img, valid_img)
# 	ssim_img = skimage.measure.compare_ssim(gt_img, valid_img, gaussian_weights=True, use_sample_covariance=False,  multichannel=True)
	
# 	avg_psnr += psnr
# 	avg_ssim += ssim_img
# 	print(psnr, ssim_img)

# avg_psnr = avg_psnr / 30
# avg_ssim = avg_ssim / 30
# print('avg psnr:', avg_psnr, avg_ssim)

#33.233 0.9060
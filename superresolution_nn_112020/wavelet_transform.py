import os
import skimage
from PIL import Image
import numpy as np
import cv2
import pywt


GT_path = '../ECCV_UDC/valid_Poled_GT'
valid_img_path = '../ECCV_UDC/valid_Poled'#'../UDC_Model3_concat/test'


for i in range(30):
	gt_img = os.path.join(GT_path, str(i) + '.png')
	valid_img = os.path.join(valid_img_path, str(i) + '.png')
	#valid_img = os.path.join(valid_img_path,'result_' + str(i+1) + '_epoch_949.png')
	print(gt_img)
	print(valid_img)

	gt_img = cv2.imread(gt_img)
	valid_img = cv2.imread(valid_img)

	coeffs2 = pywt.dwt2(valid_img[:,:,0], 'bior1.3')
	LL, (LH, HL, HH) = coeffs2

	for j, a in enumerate([LL, LH, HL, HH]):
		cv2.imwrite('input_wavelet/'+str(i*8+j*2)+'.png',a)

	coeffs2 = pywt.dwt2(gt_img[:,:,0], 'bior1.3')
	LL, (LH, HL, HH) = coeffs2

	for j, a in enumerate([LL, LH, HL, HH]):
		cv2.imwrite('input_wavelet/'+str(i*8+j*2+1)+'.png',a)


	# valid_edge = cv2.Canny(valid_img,20,50)
	# cv2.imwrite('edge/'+str(i*2)+'.png',valid_edge)

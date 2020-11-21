import numpy as np
import math
from skimage.measure import compare_ssim
from PIL import Image
import os.path
import glob


def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2, multichannel=False):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    return compare_ssim(img1, img2, multichannel=multichannel)

address = 'GT/'
address_sr= '2xdownsampled_2xsr/'
tag = ''
for path in glob.glob(address + '*'):

    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    tail = os.path.splitext(basename)[1]
    target = np.array(Image.open(address + basename).convert('RGB'))
    sr = np.array(Image.open(address + base + tag + '.png').convert('RGB'))
    PSNR = psnr(sr, target)
    SSIM = ssim(sr, target)
    print("PSNR: %f", PSNR)
    print("SSIM: %f", SSIM)



from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from dataset import *


def download_bsds300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        if not exists(dest):
            makedirs(dest)
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def get_dataset(data_dir, crop_size, scale_factor, is_gray=False):
    return TrainDatasetFromFolder(data_dir,
                                 is_gray=is_gray,
                                  crop_size = crop_size,
                                 scale_factor=scale_factor)


def get_training_DENOISE_set(data_dir, crop_size, scale_factor, is_gray=False):
    train_dir = data_dir
    return TrainDatasetFromFolder(train_dir,
                                 is_gray=is_gray,
                                  crop_size = crop_size,
                                 scale_factor=scale_factor)



def get_test_set(data_dir, scale_factor, is_gray=False):
    test_dir = data_dir
    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)

def get_test_set_LRinput(data_dir, dataset, scale_factor, is_gray=False):
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)
        test_dir = join(root_dir, "test")
    elif dataset == 'DIV2K':
        test_dir = join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4')
    else:
        ### ORIGINAL ###
        #test_dir = join(data_dir, dataset)
        ### ORIGINAL END ###

        ### DEBUG GUAN 20180102 ###
        test_dir = data_dir+'/'+dataset[0]+'/Denoise_test/'
        ### DEBUG GUAN 20180102 END ###
    return TestDatasetFromFolder_LRinput(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)


def get_vali_set(dataset, scale_factor, is_gray=False):
    vali_dir = dataset
    return TestDatasetFromFolder(vali_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)
if __name__ == '__main__':

    # args.model_name= 'FSRCNN'
    download_bsds300()
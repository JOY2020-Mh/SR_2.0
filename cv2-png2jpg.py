import os
import cv2
import sys
import numpy as np
 
#path = "/home/miaohuan/Documents/visdon_pair_training_dataset/HR/"
path = '/visdon_pair/HR_visdon'
print(path)
 
for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.png':
        print(filename)
        img = cv2.imread(path + filename)
        print(filename.replace(".png",".jpg"))
        newfilename = filename.replace(".png",".jpg")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        new_path = path + newfilename
        cv2.imwrite(new_path, img)
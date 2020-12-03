import os
import numpy as np
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os.path import join
from scipy.optimize import curve_fit
import numpy as np
from sklearn.decomposition import PCA

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()
 
def make_model():
    model=models.vgg16(pretrained=True).features[:28]
    print(models.vgg16(pretrained=True))	# 其实就是定位到第28层，对照着上面的key看就可以理解
    print(model)
    model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model
    
#特征提取
def extract_feature(model,imgpath):
    model.eval()		# 必须要有，不然会影响特征提取结果
    img=Image.open(imgpath)		# 读取图片
    #img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor=img_to_tensor(img)	# 将图片转化成tensor
    #print(tensor.shape)
    tensor = tensor.unsqueeze(0)
    #print(tensor.shape)
    #tensor=tensor.cuda()	# 如果只是在cpu上跑的话要将这行去掉
    
    result=model(Variable(tensor))
    #print('result.shape', result.shape)##torch.Size([1, 512, 1, 1])
    # result1 = result.squeeze(0)
    # print(result1.shape)
    # result2 = result1.squeeze(0)
    # print(result2.shape)
    result_npy=result.data.cpu().numpy()	# 保存的时候一定要记得转成cpu形式的，不然可能会出错

    return result_npy[0]	# 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
    
if __name__=="__main__":
    model=make_model()

    img_dir = './cropped_256_image'
    image_filenames = [join(img_dir, x) for x in sorted(listdir(img_dir))]
    
    n = len(image_filenames)
    print(n)
    ##put features to this array
    a = np.zeros((n, 131072))

    for index in range(len(image_filenames)):
        imgpath = image_filenames[index]
        tmp = extract_feature(model, imgpath)
        tmp_1 = tmp[0, :,:]
        print(tmp_1.shape)
        img = Image.fromarray(tmp_1, 'RGB')
        #img.show()
        name = str(index)+'.png'
        img.save(name)
        #print(tmp.shape) #(512* 16 * 16)
        result1 = tmp.reshape(131072)
        a[index] = result1
        # print(max(a[index]), min(a[index]))
        #print(a[index][1000:1010])
        #print(result1.shape)
        # print(max(a),min(a))
        #print(tmp.shape)	# 打印出得到的tensor的shape
        #print(tmp)		# 打印出tensor的内容，其实可以换成保存tensor的语句，这里的话就留给读者自由发挥了

    
    
    pca = PCA(n_components='mle')  
    pca.fit(a)                  #训练
    newX = pca.fit_transform(a)   #降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    print(newX.shape)
    print(pca.explained_variance_ratio_)  #输出贡献率
    print(newX)               


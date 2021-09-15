import torch
from PIL import ImageOps
from PIL import Image
from PIL import ImageFilter  # 滤波器
from torchvision.transforms import transforms
from tqdm import tqdm
import time
from torchvision.transforms import InterpolationMode
import os
import matplotlib.pyplot as plt
from numpy import *
from pylab import *
import torchvision


# im = Image.open('D:\\下载\\谷歌浏览器\\数据集\\data\\91-image\\t1.bmp')   # 打开图片

# print(im.getbands())   # 获取图片通道，返回了一个列表

# print(len(im.getbands()))    # 获取通道数

# print(im.mode)   # 获取图片的模式，返回一个字符串

# print(im.size)    # 返回图片的尺寸，分辨率

# im.show()    # 展示图片

# new_im = im.convert('L')    # 转化模式，RGB为真彩色，L为灰度图片

# new_im.save('2.jpeg')   # 保存图片，并对其进行重命名操作

# new_im = Image.new('RGB', (128,128), 'red')    # 按照给定模式，尺寸，颜色，生成一个新的图片

# new_im = im.crop((20, 30, 200, 300))    # 对图片截取一个四元组大小的区域

# new_im = im.filter(ImageFilter.CONTOUR)    # 边缘检测滤波器的使用

# r,g,b =im.split()    # 图像分离（将RGB分离成为三通道）
# im_merge = Image.merge('RGB',[r,g,b])   # 图像合成，通过给定的模式以及对应的通道的图像进行合成

# pixel = im.load()    # 返回一个像素操作对象-->print(pixel[50, 50])

# region = im.resize((30, 50))    # 将图片进行SR处理

# a = transforms.ToTensor()

# total参数设置进度条的总长度
# with tqdm(total=50) as pbar:
#   for i in range(100):
#     time.sleep(1)
#     #每次更新进度条的长度
#     pbar.update(1)

# tp1 = (1001, 1000)
# print((tp1[1], tp1[0]))
# a = transforms.Resize(tp1, InterpolationMode.BICUBIC)
# im = a(im)
# print(type(im.size))
# im.save('D:\\下载\\谷歌浏览器\\3.jpeg')
# print(os.listdir("D:\\下载\\谷歌浏览器\\数据集\\data\\91-image"))
# j = 0
# for i in os.listdir("D:\\项目\\数据集\\91-image\\train"):
#     transform = transforms.Compose([transforms.Resize((250, 250), InterpolationMode.BICUBIC),
#                                     transforms.Resize((500, 500), InterpolationMode.BICUBIC)])
#     path = "D:\\项目\\数据集\\91-image\\train\\"+i
#     im = Image.open(path)
#     new_LR = transform(im)
#     new_path = "D:\\项目\\数据集\\91-image-LR\\test_2\\"+str(j)+"_LR.bmp"
#     new_LR.save(new_path)
#     j = j+1
# filename = os.listdir("D:\\项目\\数据集\\91-image-LR\\test")
# filename.sort(key=lambda x: int(x[2:-8]))
# for i in filename:
#     print(i)
# filename = os.listdir("D:\\项目\\数据集\\91-image\\train")
# filename.sort(key=lambda x:int(x[1:-4]))
# for i in filename:
#     print(i)


def HE(path):
    im = Image.open(path)
    input = np.array(im)
    row, col = input.shape
    imhist = np.zeros((256, 1))
    for i in range(row):
        for j in range(col):
            imhist[input[i][j]] += 1
    # imhist, bins = histogram(im, 256, density=True)
    new_imhist = [0 for x in range(256)]
    for i in range(256):
        new_imhist[i] = round(255 * imhist[0:i + 1].sum())  # 计算原先的灰度值对应新的灰度值
    im = array(im)
    new_imhist = array(new_imhist)
    print(new_imhist.shape)
    im_eq = new_imhist[im]  # 关键：将新生成的直方图作用于之前的图片上生成新的图片,原理是：若b为一维的array，
    # 对于c = b[a]返回一个和a同型的array，其中，c[i][j]=b[a[i][j]]，故a[i][j]需在b的索引范围内
    print(im_eq.shape)
    a = Image.fromarray(im_eq)
    a.show()


HE("D:/项目/lesson/5.jpeg")
# im = Image.open('./base.jpeg')
# imhist, bin = histogram(im, 256)
# nvalid = 0
# for i in range(len(imhist)):
#     if imhist[i] > 0:
#         nvalid = nvalid + 1
# sum = sum(imhist)
# T = sum*0.0001
#
# for i in range(len(imhist)):
#     if imhist[i] < T:
#         imhist[i] = 0
#     else:
#         imhist[i] = 1
# B = np.array(([0 for x in range(len(imhist))]), dtype='float32')
# R = min(nvalid, 256)
# for i in range(len(imhist)):
#     if i==0:
#         B[i]=0
#     else:
#         B[i] = R * imhist[:i].sum()/nvalid
# im = np.array(im)
# im_eq = B[im]
# a = Image.fromarray(im_eq)
# a.show()


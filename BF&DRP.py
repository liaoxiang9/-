from PIL import Image
import numpy as np
from pylab import *

'''
 _______________________________________________________________________________________
|  参考文献：《New technique for the visualization of high dynamic range infrared images》 |                                                                           |
|  主要思想： 通过双边滤波器（均值滤波和高斯滤波）将图片分离成detail和base两部分，分别对两部分进行处理 |
|           然后再进行合并。对于base和detail进行gamma修正，然后进行浸润（1%），最后以一定比例进行   |
|           合并输出                                                                     | 
|                                                                                       |
|                                                                                       |
|                                                                                       |
|                                                                                       |
|______________________________________________________________________________________ |                                                                          |
'''


def BilateralFilter(img, winsize, Q=20):   # 输入一个array的图像
    row, col = img.shape
    r = int(winsize/2)
    c = r
    new_img = np.zeros((row, col))   # 初始化滤波后的图片
    max = np.max(img)
    min = np.min(img)
    sigma_g = int((max-min)/Q)
    sigma2 = 2 * sigma_g*sigma_g
    kernel = np.zeros((winsize, winsize))  # 初始化滤波器
    image1 = np.pad(img, ((r, c), (r, c)), constant_values=0)
    for i in range(r, row+r):
        for j in range(r, col+r):
            '''每一个像素都对应一个像素滤波器'''
            gkernel = np.zeros((winsize, winsize))  # 构造高斯滤波器
            for m in range(-r, r + 1):
                for n in range(-r, c + 1):
                    gkernel[m + r][n + c] = np.exp(-pow((int(image1[i][j]) - int(image1[i + m][j + n])), 2) / sigma2)
                    kernel[m + r][n + c] = (1/winsize**2) * gkernel[m + r][n + r]
            sum_kernel = sum(sum(kernel))
            kernel = kernel / sum_kernel  # 归一化
            for m in range(-r, r + 1):  # 生成新的像素
                for n in range(-c, c + 1):
                    new_img[i-r][j-r] = image1[i + m][j + n] * kernel[m + r][n + c] + new_img[i-r][j-r]
    BP = Image.fromarray(new_img)
    BP = BP.convert('L')
    BP.save('./base2.jpeg')
    return new_img


def gamma_correction(img, gamma):   # gamma修正
    row, col = img.shape
    new_img = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
                new_img[i][j] = img[i][j]*gamma*gamma
    return new_img


def saturate(input, sat):
    row, col = input.shape
    sum = row*col
    threshold = int(sat*sum)
    copy = input
    inf = np.sort(copy.flatten())[threshold-1]
    sup = np.sort(copy.flatten())[-threshold]
    for i in range(row):
        for j in range(col):
            if input[i][j] < inf:
                input[i][j] = inf
            if input[i][j] > sup:
                input[i][j] = sup
    return input


def recombine(detail, base, p):    # 这里考虑线性压缩，将原先图像范围映射至0-2^M*p
    max_d = np.max(detail)
    min_d = np.min(detail)
    row, col = detail.shape
    new_detail = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            new_detail[i][j] = ((detail[i][j] - min_d)/(max_d - min_d))*(p*256)
    max_b = np.max(base)
    min_b = np.min(base)
    new_base = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            new_base[i][j] = ((base[i][j] - min_b)/(max_b - min_b))*((1-p)*256)
    return new_base + new_detail




def show_img(img):
    BP = Image.fromarray(img)
    BP.show()


def HE(path):
    im = Image.open(path)
    imhist, bins = histogram(im, 256, density=True)
    new_imhist = [0 for x in range(256)]
    for i in range(256):
        new_imhist[i] = round(255*imhist[0:i+1].sum())   # 计算原先的灰度值对应新的灰度值
    im = np.array(im)
    new_imhist = np.array(new_imhist)
    im_eq = new_imhist[im]  # 关键：将新生成的直方图作用于之前的图片上生成新的图片,原理是：若b为一维的array，
                            # 对于c = b[a]返回一个和a同型的array，其中，c[i][j]=b[a[i][j]]，故a[i][j]需在b的索引范围内
    a = Image.fromarray(im_eq)
    a.show()


def BF_DRP(origin_img_path, winsize, Q, gamma_d, gamma_b, p):
    origin_img = Image.open(origin_img_path).convert('L')
    origin_img = np.array(origin_img)
    base = BilateralFilter(origin_img, winsize, Q)
    origin_img = np.array(origin_img, np.int32)
    base = np.array(base, np.int32)
    detail_img = origin_img - base
    col, row = detail_img.shape
    gamma_base = gamma_correction(base, gamma_b)
    gamma_detail = gamma_correction(detail_img, gamma_d)
    new_img = recombine(gamma_detail, gamma_base, p)
    new_base = saturate(gamma_base, 0.001)
    new_detail = saturate(gamma_detail, 0.001)
    new_img = recombine(new_detail, new_base, p)
    show_img(new_img)


BF_DRP("./FLIR_00780.jpeg", 3, 20, 1.2, 0.9, 0.4)
# base = Image.open("./base2.jpeg")
# origin = Image.open("./FLIR_00780.jpeg")
# im = base.convert('L')
# base_img = np.array(im, dtype='float32')
# im2 = origin.convert('L')
# origin_img = np.array(im2, dtype='int32')
# detail_img = origin_img - base_img
# col, row = detail_img.shape
# new_base = gamma_correction(base_img, 0.9)
# new_detail = gamma_correction(detail_img, 1.2)
# new_base = saturate(new_base, 0.001)
# new_detail = saturate(new_detail, 0.001)
# p = 0.4
# new_img = recombine(new_detail, new_base, p)
# show_img(new_img)
# BP = Image.fromarray(new_img)
# BP = BP.convert('L')
# BP.save('./BF.jpeg')
# BP.show()




# origin = Image.open('./FLIR_00780.jpeg')
# origin_img = np.array(origin)
# BF_DRP(origin_img, 21, 20, 2.5, 0.9, 0.1)

# a = np.array(([5, 7, 2, 9, 6, 3],
#              [1, 2, 3, 4, 5, 6]))
# s = a.flatten()
# print(np.sort(origin_img.flatten())[-100])
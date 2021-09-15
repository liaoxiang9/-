import cv2
from PIL import Image
import numpy as np
from pylab import *
import math
import os
import time
'''金字塔算法，对16bit红外图像进行增强'''
def linear(img_path):
    # time_start = time.time()
    img = Image.open(img_path)
    input = np.array(img)
    imhist = np.histogram(input, bins=65535, range=[0, 65535])[0]
    # imhist = np.zeros(65535)
    # row, col = input.shape
    # for i in range(row):
    #     for j in range(col):
    #         imhist[input[i][j]] += 1
    # time_end = time.time()
    # print("生成直方图", time_end - time_start)
    lowcut = 1000
    highcut = 1000
    # 寻找边界
    # time_start = time.time()
    minloc = 0
    sum_min = 0
    for i in range(655):
        sum_min = np.sum(imhist[:i*100])
        if sum_min >= lowcut:
            minloc = i*100
            break
    for i in range(minloc-100, minloc):
        sum_min = np.sum(imhist[:i])
        if sum_min >= lowcut:
            minloc = i
            break
    sum_hist = np.sum(imhist)
    maxloc = 65535
    sum_max = 0
    for i in range(655, -1, -1):
        sum_max = sum_hist - np.sum(imhist[:100*i])
        if sum_max >= highcut:
            maxloc = i *100
            break
    for i in range(maxloc+100, maxloc, -1):
        sum_max = sum_hist - np.sum(imhist[:i+1])
        if sum_max >= highcut:
            maxloc = i
            break
    # time_end = time.time()
    # print("寻找边界", time_end - time_start)
    # 线性变换
    # time_start = time.time()
    linear_output = np.where(input > maxloc, 65535, input)
    linear_output = np.where(linear_output < minloc, 0, linear_output)
    linear_output = np.where((linear_output>0)&(linear_output<65535), (linear_output - minloc)*(65535/(maxloc - minloc)), linear_output)
    # time_end = time.time()
    # print("线性变换", time_end - time_start)
    return linear_output

# linear('./FLIR_00001.tiff')


def resize(img, factor): #最邻近插值采样
    row, col = img.shape
    output = np.zeros((int(row*factor), int(col*factor)))
    for i in range(int(row*factor)):
        for j in range(int(col*factor)):
            output[i][j] = img[math.ceil((i+1)/factor)-1][math.ceil((j+1)/factor)-1]
    return output



def gauss(input):
    row, col = input.shape
    winsize = 5
    sigma = 5
    r = int(winsize/2)
    c = r
    kernel = np.zeros((winsize, winsize))
    for i in range(-r, r+1):
        for j in range(-c, c+1):
            kernel[i+r][j+c] = np.exp(-(i**2 + j**2)/(2 * sigma**2))
    sum_kernel = sum(sum(kernel))
    kernel = kernel/sum_kernel
    output_pad = np.pad(input, ((r, c), (r, c)), constant_values=0)
    # 对边界进行replicate操作
    for i in range(r):
        output_pad[i] = output_pad[r]
        output_pad[i+row+r] = output_pad[row+r-1]
    for i in range(c):
        output_pad[:, i] = output_pad[:, r]
        output_pad[:, i + col + c] = output_pad[:, col+c-1]
    gauss_output = np.zeros_like(input)
    # 进行高斯滤波
    for i in range(r, row + r):
        for j in range(c, col + c):
            for m in range(-r, r + 1):  # 生成新的像素
                for n in range(-c, c + 1):
                    gauss_output[i-r][j-c] += output_pad[i + m][j + n] * kernel[m + r][n + c]

    return gauss_output


def prydown(input):
    row, col = input.shape
    # 进行高斯滤波
    gauss_output = cv2.GaussianBlur(input, (5, 5), 5)
    # 进行下采样
    output = cv2.resize(gauss_output, (int(row*0.5), int(col*0.5)), cv2.INTER_NEAREST)
    return output

def pryup(input):
    row, col = input.shape
    # 进行高斯滤波
    gauss_output = cv2.GaussianBlur(input, (5, 5), 5)
    # 进行下采样
    output = cv2.resize(gauss_output, (row*2, col*2), cv2.INTER_NEAREST)
    return output


# img = Image.open('D:/项目/local/linear/FLIR_00001_linear.tiff')
# img = np.array(img)
# img = img/65535
# test = gauss(img)
# test1 = resize(test, 0.5)
# test2 = gauss(test1)
# row, col = img.shape
# g = prydown(img)
# r = pryup(g)
# # # g = np.array(g, dtype='int32')
# # # r = np.array(r, dtype='int32')
# a = img-r
# linear_img = Image.frombytes('I;16', (640, 512), a.tostring())
# linear_img.show()


def Pyramid_Enhanced(input, nums):  # nums为金字塔层数
    img = input/65535
    row, col = img.shape
    Gauss = []   # 高斯金字塔
    Laplace = []  # 拉普拉斯金字塔
    re = [0 for x in range(nums-1)]  # 重建后的金字塔
    Gauss.append(img)    # 原图放在金字塔最底层
    '''生成金字塔'''
    for i in range(1, nums):
        c = Gauss[i-1]
        R = prydown(c)  # 对img下采样
        Gauss.append(R)
        E = pryup(R)  # 对下采样图片上采样
        D = c - E    # 得到差值图片
        Laplace.append(D)
    '''对细节层进行处理'''
    a1 = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3]
    p1 = [0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.1]
    for i in range(nums-1):
        x = Laplace[i]
        temp1 = pow(np.abs(x), p1[i])
        temp2 = ((x + 0.001)/np.abs(x + 0.001)) * temp1
        temp3 = a1[i] * temp2
        Laplace[i] = temp3

    '''对顶层Gauss图做色调处理'''
    x = Gauss[nums-1]
    p = 0.75
    a = 0.75
    temp1 = pow(np.abs(x), p)
    temp2 = ((x + 0.001)/np.abs(x + 0.001)) * temp1
    temp3 = a * temp2
    re[-1] = temp3

    '''重构'''
    for i in range(nums-1):
        re_E = pryup(re[nums - 2 - i])
        E_enhance = re_E + Laplace[nums-i-2]
        if nums-2-i != 0:
            re[nums-2-i-1] = E_enhance
        else:
            img_re = E_enhance
    # img_re = cv2.GaussianBlur(img_re, (5, 5), 0)

    img_re = img_re*65535
    # for i in range(row):
    #     for j in range(col):
    #         if img_re[i][j] < 0: img_re[i][j] = 0
    #         elif img_re[i][j] > 65535: img_re[i][j] = 65535
    img_re = np.where(img_re[:, :] > 65535, 65535, img_re[:, :])
    img_re = np.where(img_re[:, :] < 0, 0, img_re[:, :])
    output = img_re.astype(np.uint16)
    output_img = Image.frombytes('I;16', (col, row), output.tostring())
    return output_img


def Pyramid_Enhance(input_path, nums):
    for input_img_name in os.listdir(input_path):
        time_start = time.time()
        img_path = os.path.join(input_path, input_img_name)
        linear_img = linear(img_path)
        pry_img = Pyramid_Enhanced(linear_img, nums)
        pry_img_name = input_img_name[:-5] + '_Enhanced_new.tiff'
        pry_img_path = os.path.join('D:/项目/local/LG_new', pry_img_name)
        pry_img.save(pry_img_path)
        time_end = time.time()
        print('totally cost:', time_end - time_start)

Pyramid_Enhance('D:/项目/local/FLIR', 5)

import numpy as np
from PIL import Image
from pylab import *
'''
 _______________________________________________________________________________________
|  参考文献：《Display and detail enhancement for high-dynamic-range infrared images》    |                                                                           |
|  主要思想： 通过双边滤波器将图片分离成detail和base两部分，分别对两部分进行处理，然后再进行合并。    |
|           对于base使用自适应高斯滤波，然后input - base = detail， 对base使用自适应直方图均衡， |
|           对detail使用自适应掩膜控制 ， 然后两部分相加得到输出                               |                     
|                                                                                       |
|                                                                                       |
|                                                                                       |
|                                                                                       |
|______________________________________________________________________________________ |                                                                          |
'''


def GaussKernel(winsize, sigma):
    '''定义空间高斯滤波器'''
    kernel = np.zeros((winsize, winsize))
    r = int(winsize/2)
    c = r
    sigma_a = 2*sigma*sigma
    for i in range(-r, r+1):
        for j in range(-c, c+1):
            kernel[i+r][j+r] = np.exp(-(i**2+j**2)/sigma_a)
    return kernel


def BilateralFilter(input, winsize, sigma_d, sigma_r,):
    '''定义双边滤波器'''
    row, col = input.shape
    r = int(winsize/2)
    c = r
    output = np.zeros((row, col))
    input_padding = np.pad(input, ((r, c), (r, c)), constant_values=0)
    kernel_d = GaussKernel(winsize, sigma_d)    # 定义空间高斯核
    sigma_r_a = 2*sigma_r*sigma_r
    gain = np.zeros((row, col))     # 初始化细节层增益系数
    for i in range(r, r+row):
        for j in range(c, col+c):
            kernel_r = np.zeros((winsize, winsize))  # 初始化像素滤波器
            kernel = np.zeros((winsize, winsize))   # 初始化最终滤波器
            for m in range(-r, r+1):
                for n in range(-c, c+1):
                    kernel_r[m+r][n+r] = np.exp(-pow((int(input_padding[i][j]) - int(input_padding[i + m][j + n])), 2)/sigma_r_a)
                    kernel[m+r][n+r] = kernel_d[m+r][n+r]*kernel_r[m+r][n+r]
            sum_kernel = sum(sum(kernel))
            gain[i-r][j-c] = sum_kernel
            kernel = kernel / sum_kernel
            for m in range(-r, r + 1):  # 生成新的像素
                for n in range(-c, c + 1):
                    output[i-r][j-c] = input_padding[i + m][j + n] * kernel[m + r][n + c] + output[i-r][j-c]
    return output, gain




def Base_Process(base, percent):
    '''这里做直方图处理，使对比度增强'''
    base = np.round(base)
    base = np.array(base, dtype='int32')
    # base_max = np.max(base)
    im = Image.fromarray(base)
    imhist, bin = histogram(im, 256)
    nvalid = 0
    for i in range(len(imhist)):
        if imhist[i] > 0:
            nvalid = nvalid + 1
    sum_pixels = np.sum(imhist)
    T = sum_pixels * percent
    for i in range(len(imhist)):
        if imhist[i] < T:
            imhist[i] = 0
        else:
            imhist[i] = 1
    B = np.array(([0 for x in range(len(imhist))]), dtype='float32')
    R = min(nvalid, 256)
    for i in range(len(imhist)):
        if i == 0:
            B[i] = 0
        else:
            B[i] = R * imhist[:i].sum() / nvalid
    new_base = B[base]
    return new_base


def second_deviation(input):
    """计算二阶导（离散形式）"""
    input_padding = np.pad(input, ((1, 1), (1, 1)), constant_values=0)
    input_padding = np.array(input_padding, dtype='int32')
    row, col = input.shape
    output = np.zeros((row, col))
    for i in range(1, row+1):
        for j in range(1, col+1):
            output[i-1][j-1] = (input_padding[i+1, j]) + input_padding[i-1][j] + input_padding[i][j+1] + input_padding[i][j-1] - 4*input_padding[i][j]
    return output



def Adaptive_Gaussian_Filter(input, bf_output, gain, winsize):
    """自适应高斯滤波器"""
    E = gain * (input - bf_output)
    second_deviarate = second_deviation(bf_output)
    row, col = bf_output.shape
    r = int(winsize/2)
    c = r
    bf_output_padding = np.pad(bf_output, ((r, c), (r, c)), constant_values=0)
    output = np.zeros((row, col))
    for i in range(r, r+row):
        for j in range(c, c+col):
            sigma = sqrt(abs((2*E[i-r][j-c]) / (second_deviarate[i-r][j-c] + 0.00001)))
            kernel = GaussKernel(winsize, sigma + 0.00001)
            kernel_sum = sum(sum(kernel))

            kernel = kernel / kernel_sum
            for m in range(-r, r + 1):  # 生成新的像素
                for n in range(-c, c + 1):
                    output[i-r][j-c] = bf_output_padding[i + m][j + n] * kernel[m + r][n + c] + output[i-r][j-c]
    return output


base = Image.open('./base.jpeg')


def Detail_Process(detail, gain, g_min, g_max):
    '''这里对细节层进行自适应增益控制'''
    # row, col = detail.shape
    # new_detail = np.zeros((row, col))
    G = g_min + (1 - gain) * (g_max - g_min)
    # for i in range(row):
    #     for j in range(col):
    #         G[i][j] = g_min + [1 - gain[i][j]]*(g_max - g_min)
    # for i in range(row):
    #     for j in range(col):
    #         new_detail[i][j] = detail[i][j]*G[i][j]
    new_detail =255 - detail * G
    return new_detail


def saturate(input, sat):
    """进行截尾浸润操作"""
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


def BF_DDE(input_path, sigma_r, winsize_bf, winsize, g_min, g_max, percent):
    """BF_DDE算法实现"""
    input_img = Image.open(input_path)
    input = np.array(input_img)
    row, col = input.shape
    sigma_d = sqrt(row**2 + col**2)*0.025
    bf_output, gain = BilateralFilter(input, winsize_bf, sigma_d, sigma_r)
    base = Adaptive_Gaussian_Filter(input, bf_output, gain, winsize)
    # base = bf_output
    input = np.array(input, dtype='float32')
    base = np.array(base, dtype='float32')
    detail = input - base
    # detail = 255 + detail
    new_base = Base_Process(base, percent)
    new_detail = Detail_Process(detail, gain, g_min, g_max)
    output = new_base + new_detail

    new_output = saturate(output, 0.01)

    max_d = np.max(new_output)
    min_d = np.min(new_output)
    if min_d >= 0 and max_d <= 255:
       new_output = new_output + (127.5 - (max_d - min_d)/2)
    else:
        new_output = ((new_output - min_d) / (max_d - min_d)) * 255
    # for i in range(row):
    #     for j in range(col):
    #         new_detail[i][j] = ((detail[i][j] - min_d) / (max_d - min_d)) * (p * 256)

    output_img = Image.fromarray(new_output)

    output_img.show()
    output_img.convert('L').save('./BF&DDE.jpeg', format='jpeg')


BF_DDE('./FLIR_00050.jpeg',  0.01, 1, 1, 1, 2.5, 0.001)
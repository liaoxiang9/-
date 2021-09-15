import numpy as np
from PIL import Image
"""引导滤波器实现"""
"""新建一个均值滤波器"""
def meanfilter(input, winsize):
    col, row = input.shape
    r = int(winsize/2)
    c = r
    output = np.zeros((col, row))
    input_pad = np.pad(input, ((r, c), (r, c)), constant_values=0)
    sum_win = winsize**2
    mean_kernel = np.ones((winsize, winsize))
    mean_kernel = mean_kernel/sum_win
    for i in range(r, col+r):
        for j in range(r, row+c):

            '''进行（i,j ）处的像素运算'''
            for m in range(-r, r+1):
                for n in range(-c, c+1):
                    output[i-r][j-c] = output[i-r][j-c] + mean_kernel[m+r][n+c]*input_pad[i+m][j+n]

        # output[i-r][j-r] = int(output[i-r][j-r])
    return output


def GuidedFilter(input, guide, winsize, epsilon):
    '''这里具体实现引导滤波器，其中I代表引导图像， p代表输入图像'''
    I_mean = meanfilter(guide, winsize)
    p_mean = meanfilter(input, winsize)
    I_corr = meanfilter(input*input, winsize)
    Ip_corr = meanfilter(guide*input, winsize)
    Var_I = I_corr - I_mean*I_mean
    Cov_Ip = Ip_corr - I_mean*p_mean
    a = Cov_Ip / (Var_I + epsilon)
    b = p_mean - a * I_mean
    a_mean = meanfilter(a, winsize)
    b_mean = meanfilter(b, winsize)
    output = a_mean * guide + b_mean
    return output


def main(img_path, r, epsilon):
    input = Image.open(img_path)
    if input.mode == 'RGB':
        input = input.convert('L')
    input_array = np.array(input, dtype='float32')
    output_array = GuidedFilter(input_array, input_array, r, epsilon)
    output = Image.fromarray(output_array)
    output.show()



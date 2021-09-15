import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

def gaus_kernel(winsize, gsigma):   # 默认高斯核为正方形的，边长为winsize
    r = int(winsize / 2)
    c = r
    kernel = np.zeros((winsize, winsize))
    sigma1 = 2 * gsigma * gsigma
    for i in range(-r, r + 1):
        for j in range(-c, c + 1):
            kernel[i + r][j + c] = np.exp(-float(float((i * i + j * j)) / sigma1))
    return kernel


def bilateral_filter(image, gsigma, ssigma, winsize):
    r = int(winsize / 2)
    c = r
    row, col = image.shape
    bilater_image = image
    image1 = np.pad(image, ((r, c), (r, c)), constant_values=0)  # 类似卷积中的padding操作，为使得生成图片的大小不变
    image = image1
    # row, col = image.shape
    sigma2 = 2 * ssigma * ssigma
    skernel = gaus_kernel(winsize, gsigma)
    kernel = np.zeros((winsize, winsize))  # 初始化滤波器
    # bilater_image = np.zeros((row, col))  # 新的，空白的图片
    for i in range(row):
        for j in range(col):
            gkernel = np.zeros((winsize, winsize))    # 构造高斯滤波器
            # 构造双边滤波器
            for m in range(-r, r + 1):
                for n in range(-c, c + 1):
                    gkernel[m + r][n + c] = np.exp(-pow((int(image[i][j]) - int(image[i + m][j + n])), 2) / sigma2)
                    kernel[m + r][n + c] = skernel[m + r][n + r] * gkernel[m + r][n + r]
            sum_kernel = sum(sum(kernel))
            kernel = kernel / sum_kernel  # 归一化
            for m in range(-r, r + 1):   # 生成新的像素
                for n in range(-c, c + 1):
                    bilater_image[i][j] = image[i + m][j + n] * kernel[m + r][n + c] + bilater_image[i][j]
    return bilater_image

im = Image.open("./2.bmp")
im = im.convert('L')
a = np.array(im)
q = (np.max(a)-np.min(a))/20
bilater_a = bilateral_filter(a, q, q, 21)
bilater = Image.fromarray(bilater_a)
bilater.convert('L').save("./7.jpeg")


def Gamma_transfer(img, sat, gamma):
    row, col = img.shape
    img1 = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            img1[i][j] = sat*pow(img[i][j], gamma)
    return img1


def ReConstruct(original_img, spatial_sigma, color_sigma, winsize, sat_B, gamma_B, sat_D, gamma_D, p):
    base = bilateral_filter(original_img, spatial_sigma, color_sigma, winsize)
    detail = original_img-base
    new_base = Gamma_transfer(base, sat_B, gamma_B)
    new_detail = Gamma_transfer(detail, sat_D, gamma_D)
    new_img = p*new_detail + (1-p)*new_base
    return new_img


im = Image.open("./2.bmp")
im = im.convert('L')
im = np.array(im)
q = (np.max(im)-np.min(im))/20
# base = cv2.bilateralFilter(im, 21, q, q)
# bilater = Image.fromarray(base)
# bilater.save('./base.jpeg')
base = Image.open("./base.jpeg")
base = np.array(base)
# base = base.convert('L')
# base = np.array(base)
detail = im - base
# bilater = Image.fromarray(detail)
# bilater.show()
new_base = Gamma_transfer(base, 1, 0.9)
# bilater = Image.fromarray(new_base)
# bilater.show()
new_detail = Gamma_transfer(detail, 1, 1.3)
p = 0.4
new_img = p*new_detail + (1-p)*new_base
bilater = Image.fromarray(new_img)
bilater.show()

# bilateral_filter_img1 = cv2.bilateralFilter(im, 9, 75, 75)
# bilater = Image.fromarray(bilateral_filter_img1)

import numpy as np
from PIL import Image
import cv2
import time
'''Retinex算法实现，多尺度，进行图像增强，去雾'''


def adapt_equalhist(image):
    #clipLimit参数表示对比度的大小。
    #tileGridSize参数表示每次处理块的大小
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    dst = clahe.apply(image)
    return dst
def Retinex_MSR(src_path, is_HE=False):
    ssr_img = Image.open(src_path)
    ssr_img = np.array(ssr_img, dtype=np.float32)
    row, col = ssr_img.shape[:2]
    R = cv2.split(ssr_img)[0]
    G = cv2.split(ssr_img)[1]
    B = cv2.split(ssr_img)[2]
    '''对R通道进行增强'''
    Rlog = cv2.log(R+1)
    Rfft2 = np.fft.fft2(R)
    sigma1 = 128
    F1 = np.matmul(cv2.getGaussianKernel(row, sigma1), cv2.getGaussianKernel(col, sigma1).transpose())
    EFFt1 = np.fft.fft2(F1)
    DR0 = Rfft2 * EFFt1
    DR = np.abs(np.fft.ifft2(DR0))
    DRlog = cv2.log(DR+1)
    Rr1 = Rlog - DRlog
    sigma2 = 256
    F2 = np.matmul(cv2.getGaussianKernel(row, sigma2), cv2.getGaussianKernel(col, sigma2).transpose())
    EFFt2 = np.fft.fft2(F2)
    DR0 = Rfft2 * EFFt2
    DR = np.abs(np.fft.ifft2(DR0))
    DRlog = cv2.log(DR + 1)
    Rr2 = Rlog - DRlog
    sigma3 = 512
    F3 = np.matmul(cv2.getGaussianKernel(row, sigma3), cv2.getGaussianKernel(col, sigma3).transpose())
    EFFt3 = np.fft.fft2(F3)
    DR0 = Rfft2 * EFFt3
    DR = np.abs(np.fft.ifft2(DR0))
    DRlog = cv2.log(DR + 1)
    Rr3 = Rlog - DRlog
    Rr = (Rr1 + Rr2 + Rr3)/3
    a = 125
    II = R + G + B
    Ir = R * a
    C = Ir/(II+0.00001)
    C = cv2.log(C+1)
    Rr = C * Rr
    EXPRr = cv2.exp(Rr)
    MIN = np.min(EXPRr)
    MAX = np.max(EXPRr)
    EXPRr = np.uint8(((EXPRr - MIN)/(MAX - MIN))*255)
    # if is_HE: EXPRr = cv2.equalizeHist(EXPRr)
    if is_HE: EXPRr = adapt_equalhist(EXPRr)

    '''对G通道进行增强'''
    Glog = cv2.log(G + 1)
    Gfft2 = np.fft.fft2(G)
    DG0 = Gfft2 * EFFt1
    DG = np.abs(np.fft.ifft2(DG0))
    DGlog = cv2.log(DG + 1)
    Gg1 = Glog - DGlog
    DG0 = Gfft2 * EFFt2
    DG = np.abs(np.fft.ifft2(DG0))
    DGlog = cv2.log(DG + 1)
    Gg2 = Glog - DGlog
    DG0 = Gfft2 * EFFt3
    DG = np.abs(np.fft.ifft2(DG0))
    DGlog = cv2.log(DG + 1)
    Gg3 = Glog - DGlog
    Gg = (Gg1 + Gg2 + Gg3) / 3
    Ig = G * a
    C = Ig / (II+0.00001)
    C = cv2.log(C + 1)
    Gg = C * Gg
    EXPGg = cv2.exp(Gg)
    MIN = np.min(EXPGg)
    MAX = np.max(EXPGg)
    EXPGg = np.uint8(((EXPGg - MIN) / (MAX - MIN)) * 255)
    # if is_HE: EXPGg = cv2.equalizeHist(EXPGg)
    if is_HE: EXPGg = adapt_equalhist(EXPGg)

    '''对B通道进行增强'''
    Blog = cv2.log(B + 1)
    Bfft2 = np.fft.fft2(B)
    DB0 = Bfft2 * EFFt1
    DB = np.abs(np.fft.ifft2(DB0))
    DBlog = cv2.log(DB + 1)
    Bb1 = Blog - DBlog
    DB0 = Bfft2 * EFFt2
    DB = np.abs(np.fft.ifft2(DB0))
    DBlog = cv2.log(DB + 1)
    Bb2 = Blog - DBlog
    DB0 = Gfft2 * EFFt3
    DB = np.abs(np.fft.ifft2(DB0))
    DBlog = cv2.log(DB + 1)
    Bb3 = Blog - DBlog
    Bb = (Bb1 + Bb2 + Bb3) / 3
    Ib = B * a
    C = Ib / (II+0.00001)
    C = cv2.log(C + 1)
    Bb = C * Bb
    EXPBb = cv2.exp(Bb)
    MIN = np.min(EXPBb)
    MAX = np.max(EXPBb)
    EXPBb = np.uint8(((EXPBb - MIN) / (MAX - MIN)) * 255)
    # if is_HE: EXPBb = cv2.equalizeHist(EXPBb)
    if is_HE: EXPBb = adapt_equalhist(EXPBb)
    output = cv2.merge([EXPRr, EXPGg, EXPBb])
    output_img = Image.fromarray(output)
    output_img.show()


'''经实验，MSR对强光抑制效果较好'''
time_start = time.time()
Retinex_MSR('D:\项目\强光抑制\强光抑制2.jpg', True)
time_end = time.time()
print("总计用时：", time_end-time_start)
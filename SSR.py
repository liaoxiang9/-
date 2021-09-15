import numpy as np
from PIL import Image
import cv2
import time
'''Retinex算法实现，单尺度，进行图像增强，去雾'''

def adapt_equalhist(image):
    #clipLimit参数表示对比度的大小。
    #tileGridSize参数表示每次处理块的大小
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(20, 20))
    dst  = clahe.apply(image)
    return dst



def channel_process(R,row,col, is_HE):
    time_s_1 = time.time()
    Rlog = cv2.log(R + 1)
    # Rfft2 = np.fft.fft2(R)
    sigma = 25
    F = np.matmul(cv2.getGaussianKernel(row, sigma), cv2.getGaussianKernel(col, sigma).transpose())
    # EFFt = np.fft.fft2(F)
    # DR0 = Rfft2 * EFFt
    # DR = np.abs(np.fft.ifft2(DR0))
    # DRlog = cv2.log(DR + 1)
    # a = cv2.GaussianBlur(R, (0, 0), sigma)
    # DRlog = cv2.log(cv2.GaussianBlur(R, (0, 0), sigma)+1)
    DRlog = cv2.log(F * R + 1)
    Rr = Rlog - DRlog
    EXPRr = cv2.exp(Rr)
    EXPRr = np.power(EXPRr, 1)
    MIN = np.min(EXPRr)
    MAX = np.max(EXPRr)
    EXPRr = np.uint8(((EXPRr - MIN) / (MAX - MIN)) * 255)
    # EXPRr = np.uint8(np.minimum(np.maximum(EXPRr, 0), 255))
    if is_HE: EXPRr = adapt_equalhist(EXPRr)
    time_e = time.time()
    print(time_e - time_s_1)
    return EXPRr

def Retinex_SSR(src_path, is_HE=False):
    ssr_img = Image.open(src_path)
    ssr_img = np.array(ssr_img, dtype=np.float32)
    row, col = ssr_img.shape[:2]
    R = cv2.split(ssr_img)[0]
    EXPRr = channel_process(R, row, col, is_HE)


    G = cv2.split(ssr_img)[1]
    EXPGg = channel_process(G, row, col, is_HE)

    B = cv2.split(ssr_img)[2]
    EXPBb = channel_process(B, row, col, is_HE)

    output = cv2.merge([EXPRr, EXPGg, EXPBb])
    output_img = Image.fromarray(output)
    output_img.show()

# SSR
def value_process(input_path):
    img = cv2.imread(input_path)
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = HSV_img[2]
    v_ = cv2.medianBlur(v, 3)
    l = cv2.log()



'''经实验，SSR对弱光增强以及去雾效果较好'''
time_start = time.time()
# Retinex_SSR('D:\项目\强光抑制\\强光抑制3.jpg', True)
# Retinex_SSR('D:\项目\暗光增强\\暗光增强2.jpg', True)
Retinex_SSR('D:\项目\去雾\\去雾2.jpg', False)
# Retinex_SSR('test2.jpeg',True)
time_end = time.time()
print("总计用时：", time_end-time_start)

# time_start = time.time()
# SSR('D:\项目\暗光增强\暗光增强2.jpg', 5)
# time_end = time.time()
# print("总计用时：", time_end-time_start)

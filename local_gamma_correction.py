import cv2
import numpy as np
import math


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def nothing(x):
    pass


file_path = "D:\\DataSet\\test2.jpg"
# img_gray = cv2.imread(file_path, 0)  # 灰度图读取，用于计算gamma值
img = cv2.imread(file_path)  # 原图读取
v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
v_ = cv2.GaussianBlur(v, (3, 3), 1)
# v_ = cv2.blur(v, (11,11))
v_ = 0.5 + (v_/255)*0.3
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]
r = np.power(r, v_)
MIN = np.min(r)
MAX = np.max(r)
r = np.uint8(((r - MIN) / (MAX - MIN)) * 255)
g = np.power(g, v_)
MIN = np.min(g)
MAX = np.max(g)
g = np.uint8(((g - MIN) / (MAX - MIN)) * 255)
b = np.power(b, v_)
MIN = np.min(b)
MAX = np.max(b)
b = np.uint8(((b - MIN) / (MAX - MIN)) * 255)
img = cv2.merge([r, g, b])


cv2.imshow('dsadas', img)
# mean = np.mean(img_gray)
# gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
#
# image_gamma_correct = gamma_trans(img, gamma_val)  # gamma变换
#
# # print(mean,np.mean(image_gamma_correct))
#
# cv2.imshow('image_raw', img)
# cv2.imshow('image_gamma', image_gamma_correct)
cv2.waitKey(0)

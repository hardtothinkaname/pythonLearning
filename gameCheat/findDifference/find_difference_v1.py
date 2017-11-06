# QQ游戏，大家来找茬
# 思路：简单的用opencv。
# 1.首先找出两张图片的区域。
# 2.找出两幅图片的不同，输出坐标。
# 3.使用鼠标模拟器（按键模拟器来）模拟点击事件，进行点击

import cv2
from matplotlib import pyplot as plt
import numpy as np

# 先对图片做一些简单的预处理
# 读出预先准备的图片
img = cv2.imread(r'D:\pythonProject\pythonLearning\gameCheat\findDifference\picSource\pic3.PNG', cv2.IMREAD_COLOR)
# 转换图片的格式，从BGR转换为HSV
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 进行一些简单的滤波
img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# 找两张图片的位置
# 二值化
img_gray_blur_threshold = cv2.adaptiveThreshold(img_gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# 区域连通，区分object
_, labels, stats, _ = cv2.connectedComponentsWithStats(img_gray_blur_threshold)
# 找出两个图形的区域，这里的筛选条件为，面积为60000-63000
# 注，这里的筛选条件对于不同的屏幕分辨率会有偏差。如果能如果使用后面三个参数同时匹配会更好，即，长宽面积相近时
regions = [x for x in stats if 400 > x[2] > 380]


region_width = min(regions[0][2], regions[1][2])
region_height = min(regions[0][3], regions[1][3])

regions1 = regions[0]
img_detect1 = img_gray_blur[regions1[1]:regions1[1] + region_height, regions1[0]:regions1[0] + region_width]
regions1 = regions[1]
img_detect2 = img_gray_blur[regions1[1]:regions1[1] + region_height, regions1[0]:regions1[0] + region_width]

img_detect1_int16 = np.int16(img_detect1)
img_detect2_int16 = np.int16(img_detect2)

# img_subtract = np.subtract(img_detect1, img_detect2)
img_subtract = np.subtract(img_detect1_int16, img_detect2_int16)
img_subtract_abs = np.abs(img_subtract)
img_subtract_abs_blur = cv2.blur(img_subtract_abs, (7, 7))

print('regions:', regions)
plt.imshow(img_subtract_abs)
plt.show()

# 做到这里，还是不怎么满意，发现有一些不应该出现的检测区域出现了，过检了。
# 怀疑是因为模糊的问题所以过检了。现在做第二版本

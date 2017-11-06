# QQ游戏，大家来找茬
# 思路：简单的用opencv。
# 1.首先找出两张图片的区域。
# 2.找出两幅图片的不同，输出坐标。
# 3.使用鼠标模拟器（按键模拟器来）模拟点击事件，进行点击

# v2经过上一版本，有一个初步的程序了，但是误检较为严重，所以，现在尝试去除滤波进行测试
# v3 现一版本需要直接多屏幕截图，并且检测
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageGrab

# 先对图片做一些简单的预处理
# 读出预先准备的图片
# img = np.array(ImageGrab.grab())
img = cv2.imread(r'D:\pythonProject\pythonLearning\gameCheat\findDifference\picSource\pic5.PNG', cv2.IMREAD_COLOR)
# 转换图片的格式，从BGR转换为HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
_, _, img_value = cv2.split(img_hsv)
# 进行一些简单的滤波
img_gray_blur = cv2.GaussianBlur(img_value, (5, 5), 0)

# 找两张图片的位置
# 二值化
# 因为前期发现效果太差，不使用滤波后的图片进行二值化
img_gray_blur_threshold = cv2.adaptiveThreshold(img_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# 区域连通，区分object
_, labels, stats, _ = cv2.connectedComponentsWithStats(img_gray_blur_threshold)
# 找出两个图形的区域，这里的筛选条件为，面积为60000-63000
# 注，这里的筛选条件对于不同的屏幕分辨率会有偏差。如果能如果使用后面三个参数同时匹配会更好，即，长宽面积相近时
regions = [x for x in stats if 400 > x[2] > 380]


region_width = min(regions[0][2], regions[1][2])
region_height = min(regions[0][3], regions[1][3])

regions1 = regions[0]
img_detect1 = img[regions1[1]:regions1[1] + region_height, regions1[0]:regions1[0] + region_width]
regions1 = regions[1]
img_detect2 = img[regions1[1]:regions1[1] + region_height, regions1[0]:regions1[0] + region_width]

img_detect1_int16 = np.int16(img_detect1)
img_detect2_int16 = np.int16(img_detect2)

# img_subtract = np.subtract(img_detect1, img_detect2)
img_subtract = np.subtract(img_detect1_int16, img_detect2_int16)
img_subtract_abs = np.uint8(np.abs(img_subtract))

img_subtract_abs_gray = cv2.cvtColor(img_subtract_abs, cv2.COLOR_BGR2GRAY)

# 没想到一把滤波去除效果好了很多。
# 现在把高这的区域选出来。
_, dst = cv2.threshold(img_subtract_abs_gray, 30, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
dst_closing = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

# 把周边的线条清除
rect = np.ones((region_height - 20, region_width - 20), np.uint8) * 255
rect_zero = np.zeros((region_height, region_width), np.uint8)
rect_with_border = cv2.rectangle(rect_zero, (5, 5), (region_width - 5, region_height - 5), 255, -1)
rect_without_border = cv2.bitwise_and(rect_with_border, dst_closing,)

# print('regions:', regions)
plt.imshow(img_subtract_abs)
# plt.subplot(121)
# plt.imshow(img_detect1_int16)
# plt.subplot(122)
# plt.imshow(img_detect2_int16)
plt.show()

# 因为一开始的时候是，将图片转换成graY形式来处理的，这样就分别不了红绿色。
# 现在做一下调整，调整后的程序如上。
# 如果进行数据筛选

_, _, last_stats, last_centroids = cv2.connectedComponentsWithStats(rect_without_border)
last_centroids = np.int16(last_centroids)
obj_msg_all = np.column_stack((last_stats, last_centroids))
obj_msg_select = [x for x in obj_msg_all if x[4] > 100 and x[0] > 1 and x[1] > 1]
obj_msg_select_sorted = sorted(obj_msg_select, key=lambda obj_single: obj_single[4], reverse=True)

# 增加偏移坐标
bias = [0, 0, 0, 0, 0, regions1[0], regions1[1]]
obj_msg_select_sorted_add_bias = np.add(obj_msg_select_sorted, bias)
# print(obj_msg_select)
print(obj_msg_select_sorted)
print(obj_msg_select_sorted_add_bias)

show_pic = False
if show_pic:

    for obj_rect in obj_msg_select_sorted_add_bias:
        cv2.rectangle(img_detect2, (obj_rect[0], obj_rect[1]), (obj_rect[0] + obj_rect[2], obj_rect[1] + obj_rect[3]), [0, 0, 255])
    img_detect2 = cv2.cvtColor(img_detect2, cv2.COLOR_BGR2RGB)
    plt.imshow(img_detect2)
    plt.show()

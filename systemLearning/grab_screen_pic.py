import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageGrab

img = ImageGrab.grab()

img_arr = np.array(img)

# img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)



print(type(img_arr))


plt.imshow(img_arr)
plt.show()

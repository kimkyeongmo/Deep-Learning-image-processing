#-*- coding:utf-8 -*-
import cv2
import os
from matplotlib import pyplot as plt # as는 alias 적용시 사용

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = '../image/data-jpg'
fname = os.path.join(path, "starry_night.jpg")

img = cv2.imread(fname, cv2.IMREAD_COLOR)

plt.imshow(img)
plt.xticks([]) # x축 눈금
plt.yticks([]) # y축 눈금
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


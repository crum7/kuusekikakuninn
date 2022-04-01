import torch
import cv2
import numpy as np


IMAGE_PATH = '/home/snowyowl/Downloads/yolov5-master/data/images/252532.jpg'

cv2_back_img = cv2.imread(IMAGE_PATH)
#ブランク画像
height,width = cv2_back_img.shape[:2]
print(height,width)
blank = np.zeros((height, width,1))
blank += 255 #←全ゼロデータに255を足してホワイトにする
 
cv2.imwrite('/home/snowyowl/Documents/opencv/blank.jpg',blank)
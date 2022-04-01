import torch
import cv2
import numpy as np

#定数
IMAGE_PATH = '/home/snowyowl/Downloads/yolov5-master/data/images/IMG_3314.JPG'

#yolov5を動かす runs/train/exp80/weights/best.pt
model = torch.hub.load('/home/snowyowl/Downloads/yolov5-master', 'custom', path='/home/snowyowl/Downloads/yolov5-master/runs/train/exp80/weights/best.pt', source='local')

 
results = model(IMAGE_PATH)
v = (results.display(pprint=True))




#画像の定義
desk_img = cv2.imread('/home/snowyowl/Documents/opencv/desk.jpg')
person_img = cv2.imread('/home/snowyowl/Documents/opencv/シンプルな人のピクトグラム.jpg')


#back_img = cv2.imread(IMAGE_PATH)
back_img = cv2.imread('/home/snowyowl/Documents/opencv/blank.jpg')
cv2_back_img = cv2.imread(IMAGE_PATH)
h,w = cv2_back_img.shape[:2]




for i in range(len(results.pandas().xyxy[0])):
 
   if results.pandas().xyxy[0]["class"][i] == 1:
 
 
 
 
       #座標を取得
       for j in range(len(results.pandas().xyxy[0])):
 
           if results.pandas().xyxy[0]["class"][j] == 1:
               ymin_person = int(results.pandas().xyxy[0]["ymin"][j])
               ymax_person = int(results.pandas().xyxy[0]["ymax"][j])
               xmin_person = int(results.pandas().xyxy[0]["xmin"][j])
               xmax_person = int(results.pandas().xyxy[0]["xmax"][j]) 
  
 
 
 
           #画像の位置は、ここで変化させる
               dx_person = xmin_person
               dy_person = ymin_person
 
           #画像の幅と高さを変更
               re_person_img = cv2.resize(person_img, (xmax_person-xmin_person, ymax_person-ymin_person))
 
           #平行移動するので、np.array内の数字はいじる必要なし
               M = np.array([[1, 0, dx_person], [0, 1, dy_person]], dtype=float)
               img_warped_person = cv2.warpAffine(re_person_img, M, (w, h), back_img, borderMode=cv2.BORDER_TRANSPARENT)
 
 
 
 
 
 
   if results.pandas().xyxy[0]["class"][i] == 0:
        for k in range(len(results.pandas().xyxy[0])):
 
           if results.pandas().xyxy[0]["class"][k] == 0:
               ymin_desk = int(results.pandas().xyxy[0]["ymin"][k])
               ymax_desk = int(results.pandas().xyxy[0]["ymax"][k])
               xmin_desk = int(results.pandas().xyxy[0]["xmin"][k])
               xmax_desk = int(results.pandas().xyxy[0]["xmax"][k]) 
  
 
 
 
           #画像の位置は、ここで変化させる
               dx_desk = xmin_desk
               dy_desk = ymin_desk
 
           #画像の幅と高さを変更
               re_desk_img = cv2.resize(desk_img, (xmax_desk - xmin_desk, ymax_desk - ymin_desk))
               print(xmax_desk-xmin_desk, ymax_desk-ymin_desk)
 
           #平行移動するので、np.array内の数字はいじる必要なし
               N = np.array([[1, 0, dx_desk], [0, 1, dy_desk]], dtype=float)
               img_warped_desk = cv2.warpAffine(re_desk_img, N, (w, h), back_img, borderMode=cv2.BORDER_TRANSPARENT)

        




#いつもの
img_resize = cv2.resize(back_img,(1200,900))
cv2.imshow('image',img_resize)
cv2.waitKey(0)
cv2.destroyAllwindows()


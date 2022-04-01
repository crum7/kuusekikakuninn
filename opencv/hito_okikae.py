import torch
import cv2
import numpy as np

#定数
IMAGE_PATH = '/home/snowyowl/Downloads/yolov5-master/data/images/252532.jpg'

#yolov5を動かす
model = torch.hub.load('/home/snowyowl/Downloads/yolov5-master', 'custom', path='/home/snowyowl/Downloads/yolov5-master/runs/train/exp80/weights/best.pt', source='local')


results = model(IMAGE_PATH)
v = (results.display(pprint=True))
print(v)

#置き換え処理
fore_img = cv2.imread('/home/snowyowl/Documents/opencv/シンプルな人のピクトグラム.jpg')
back_img = cv2.imread(IMAGE_PATH)
h,w = back_img.shape[:2]
print(results.pandas().xyxy[0])


#'person'がない場合は、写真を表示しない処理
for i in range(len(results.pandas().xyxy[0])):

    if results.pandas().xyxy[0]["class"][i] == 1:




        #座標を取得
        for j in range(len(results.pandas().xyxy[0])):

            if results.pandas().xyxy[0]["class"][j] == 1:
                ymin = int(results.pandas().xyxy[0]["ymin"][j])
                ymax = int(results.pandas().xyxy[0]["ymax"][j])
                xmin = int(results.pandas().xyxy[0]["xmin"][j])
                xmax = int(results.pandas().xyxy[0]["xmax"][j])  
    
                print(j)
                print(ymin)


            #画像の位置は、ここで変化させる
                dx = xmin
                dy = ymin

            #画像の幅と高さを変更
                re_fore_img = cv2.resize(fore_img, (xmax-xmin, ymax-ymin))

            #平行移動するので、np.array内の数字はいじる必要なし
                M = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
                img_warped = cv2.warpAffine(re_fore_img, M, (w, h), back_img, borderMode=cv2.BORDER_TRANSPARENT)



            #いつもの
        img_resize = cv2.resize(back_img,(1200,900))
        cv2.imshow('image',img_resize)
        cv2.waitKey(0)
        cv2.destroyAllwindows()


    else:
        quit()







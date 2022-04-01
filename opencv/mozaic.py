import torch
import cv2

#定数
IMAGE_PATH = '/home/snowyowl/Downloads/yolov5-master/data/images/IMG_3297.JPG'

#yolov5を動かす
model = torch.hub.load('/home/snowyowl/Downloads/yolov5-master', 'custom', path='/home/snowyowl/Downloads/yolov5-master/runs/train/exp80/weights/best.pt', source='local')

img = [IMAGE_PATH] 
results = model(img)

#座標を取得
for j in range(len(results.pandas().xyxy[0])):

               if results.pandas().xyxy[0]["class"][j] == 1:
                    ymin = int(results.pandas().xyxy[0]["ymin"][j])
                    ymax = int(results.pandas().xyxy[0]["ymax"][j])
                    xmin = int(results.pandas().xyxy[0]["xmin"][j])
                    xmax = int(results.pandas().xyxy[0]["xmax"][j])  
print(xmin,ymin,xmax,ymax)                       


#モザイク処理
src = cv2.imread(IMAGE_PATH)


def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

#mozaic(src, ratio)のratioの数字を小さくすればするほど、よりわからなくなる。
def mosaic_area(src, x, y, width, height, ratio=0.01):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

#先程の座標
dst_area = mosaic_area(src, xmin, ymin, xmax-xmin, ymax-ymin)

#いつもの
img_resize = cv2.resize(dst_area,(1200,900))
cv2.imshow('image',img_resize)
cv2.waitKey(0)
cv2.destroyAllwindows()
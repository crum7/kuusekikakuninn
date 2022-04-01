import cv2

'''   "cv2.imread()"は、画像を読み込む  
第１引数には、「画像ファイル」、第２引数には「フラグを設定」することができる。  
cv2.IMREAD_COLOR:カラー画像として読み込む。デフォルト
cv2.IMREAD_GRAYSCALE:グレースケール画像として読み込む
cv2.IMREAD_UNCHANGED:アルファチャンネル(.pngとかの透明度)も含めた画像として読み込む
'''


img = cv2.imread('/home/snowyowl/Downloads/yolov5-master/runs/detect/exp61/IMG_3297.JPG')


img_resize = cv2.resize(img,(400,300))

'''   "cv2.imshow()"は、画像をウインドウ上に表示する
第1引数には、「文字列型で指定するウインドウ名」、第2引数には「表示したい画像」
複数のウインドウを表示させることができるが、各ウインドウには異なる名前をつけないといけない。
日本語の名前は、禁止
'''
cv2.imshow('image',img_resize)
cv2.waitKey(0)
cv2.destroyAllwindows()


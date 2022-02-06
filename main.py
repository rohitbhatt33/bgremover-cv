import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
img = cv2.VideoCapture(0)
img.set(3, 640)
img.set(4, 480)
img.set(cv2.CAP_PROP_FPS, 60)
seg = SelfiSegmentation()
fpsread = cvzone.FPS()

lists = os.listdir("img")
print(lists)
imglist = []
for imp in lists:
    jk = cv2.imread(f'img/{imp}')
    imglist.append(jk)
print(len(imglist))

index = 0

while True:
    suc, jk = img.read()
    s = seg.removeBG(jk, imglist[index], threshold=0.67)
    k = cvzone.stackImages([jk, s], 2, 1)
    _, k = fpsread.update(k, color=(255, 0, 255))
    cv2.imshow('img', k)
    print(index)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if index > 0:
            index -= 1
    if key == ord('s'):
        if index < len(imglist)-1:
            index += 1
    if key == ord('q'):
        break

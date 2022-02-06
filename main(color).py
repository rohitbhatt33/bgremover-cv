import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation
img = cv2.VideoCapture(0)
img.set(3, 640)
img.set(4, 480)
img.set(cv2.CAP_PROP_FPS, 60)
seg = SelfiSegmentation()
fpsread = cvzone.FPS()

while True:
    suc, jk = img.read()
    s = seg.removeBG(jk, (255, 0, 0), threshold=0.67)
    k = cvzone.stackImages([jk, s], 2, 1)
    _, k = fpsread.update(k, color=(255, 0, 255))
    cv2.imshow('img', k)
    if cv2.waitKey(20) == ord('q'):
        break

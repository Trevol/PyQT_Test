import cv2
import numpy as np

def intt(iterable):
    return tuple(round(int(i)) for i in iterable)


im = np.zeros((100, 100), dtype=np.uint8)

cv2.ellipse(im, intt((50, 50.56)), intt((30.78, 20)), 15.78, startAngle=0, endAngle=360, color=(255, 0, 0), thickness=1)

cv2.imshow('123', im)
cv2.waitKey()

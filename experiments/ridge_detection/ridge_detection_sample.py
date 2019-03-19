# ndarray to pixmap
# https://www.swharden.com/wp/2013-06-03-realtime-image-pixelmap-from-numpy-array-data-in-qt/
import sys_excepthook_setup
import numpy as np
import cv2
from time import time
from skimage.morphology import skeletonize
from skimage import img_as_float, img_as_ubyte

frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
img = cv2.imread(frame)

img = cv2.blur(img, ksize=(3, 3))
#img = cv2.medianBlur(img, ksize=3)

f = cv2.ximgproc.RidgeDetectionFilter_create(dx=1, dy=1)
ridges = f.getRidgeFilteredImage(img)
_, ridges = cv2.threshold(ridges, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
#ridges = skeletonize(img_as_float(ridges))

cv2.imshow('blur', img_as_ubyte(ridges)), cv2.waitKey()

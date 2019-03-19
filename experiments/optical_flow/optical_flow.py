import cv2 as cv
import numpy as np
import video_sources
import utils
import math

cap = cv.VideoCapture(video_sources.video_2)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
max_x = max_y = -np.Infinity
min_x = min_y = np.Infinity

while (1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    _min_x = flow[..., 0].min()
    _max_x = flow[..., 0].max()
    _min_y = flow[..., 1].min()
    _max_y = flow[..., 1].max()
    if _max_x > max_x:
        max_x = _max_x
    if _min_x < min_x:
        min_x = _min_x
    if _max_y > max_y:
        max_y = _max_y
    if _min_y < min_y:
        min_y = _min_y

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('frame2', bgr)
    cv.imshow('original', frame2)

    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

print((min_x, max_x), (min_y, max_y))
cap.release()
cv.destroyAllWindows()

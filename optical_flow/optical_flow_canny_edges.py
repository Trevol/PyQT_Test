import cv2 as cv
import numpy as np

def blur(im):
    return cv.blur(im, ksize=(3, 3))

def edges(gray):
    gray = blur(gray)
    edges = cv.Canny(gray, 255 * 0.00, 255 * 1.00, edges=None, apertureSize=3, L2gradient=False)
    return edges * 255


cap = cv.VideoCapture("d:\DiskE\Computer_Vision_Task\Video 2.mp4")
ret, frame1 = cap.read()

frame1 = blur(frame1)

prvs = edges(cv.cvtColor(frame1,cv.COLOR_BGR2GRAY))

hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()

    frame2 = blur(frame2)

    next = edges(cv.cvtColor(frame2,cv.COLOR_BGR2GRAY))
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    cv.imshow('frame2',bgr)
    cv.imshow('original', frame2)

    k = cv.waitKey(int(1000 // 15)) & 0xff
    if k == 27:
        break
    prvs = next
cap.release()
cv.destroyAllWindows()
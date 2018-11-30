import numpy as np
import cv2 as cv
import time


def do_substraction(video_source, bg_subtractor, delay = None):
    cap = cv.VideoCapture(video_source)
    while (1):
        ret, frame = cap.read()

        t0 = time.time()
        fgmask = bg_subtractor.apply(frame)
        duration = time.time() - t0
        #print(np.unique(fgmask))
        putFrameNum(fgmask, duration, cap)

        cv.imshow('frame', fgmask)
        k = cv.waitKey(delay or 0) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

def putFrameNum(frame, duration, video):
    framePos = int(video.get(cv.CAP_PROP_POS_FRAMES))
    txt = f'{framePos}  {duration:.2f}'
    cv.putText(frame, txt, (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (155, 155, 155), thickness=2)
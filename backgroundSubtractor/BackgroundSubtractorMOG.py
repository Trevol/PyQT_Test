import numpy as np
import cv2 as cv


def main():
    cap = cv.VideoCapture("d:\DiskE\Computer_Vision_Task\Video 2.mp4")
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    while (1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        putFrameNum(fgmask, cap)

        cv.imshow('frame', fgmask)
        k = cv.waitKey() & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

def putFrameNum(frame, video):
    framePos = int(video.get(cv.CAP_PROP_POS_FRAMES))
    cv.putText(frame, str(framePos), (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (155, 155, 155), thickness=2)

main()
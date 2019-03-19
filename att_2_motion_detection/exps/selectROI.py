import numpy as np
import cv2
from cv_named_window import CvNamedWindow, cvWaitKeys


def main():
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 75, (0, 180, 0), -1)
    wnd = CvNamedWindow('select ROI')
    while True:
        wnd.imshow(img)
        key = cvWaitKeys(27, ord('r'), ord('s'))
        if key in (27, -1):
            break


if __name__ == '__main__':
    main()

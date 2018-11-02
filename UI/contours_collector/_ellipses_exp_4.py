from UI.contours_collector.contours_collector import ContoursCollector
import cv2
import numpy as np


def main():
    img = np.zeros((300, 300), dtype=np.uint8)

    cv2.ellipse(img, (100, 100), (60, 80), 15, 0, 260, 255, 1)
    #cv2.circle(img, (100, 100), 60, 255, 1)
    #cv2.rectangle(img, (75, 75), (125, 125), 0, -1)

    _, contours, _ = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    for contour in contours:
        contour = contour.reshape((contour.shape[0], contour.shape[2]))  # (x, 1, y) -> (x, y)
        print('  ', len(contour))

        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_bgr, [contour], -1, (255, 0, 0), 1)
        cv2.circle(img_bgr, tuple(contour[0]), 2, (0, 0, 255), -1)
        cv2.circle(img_bgr, tuple(contour[-1]), 2, (0, 255, 0), -1)

        # animate point from 0 to -1
        for point in contour:
            im = img_bgr.copy()
            cv2.circle(im, tuple(point), 3, (0, 0, 255), 1)
            cv2.imshow('ccc', im)
            if cv2.waitKey() == 27:
                return

    while cv2.waitKey() != 27:
        pass


main()

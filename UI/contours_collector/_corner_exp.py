from UI.contours_collector.contours_collector import ContoursCollector
import cv2
import numpy as np


def imread(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"

    # 806-288  1099 379
    cc = ContoursCollector(imread(frame)[288:379, 806:1099])

    cc.find_contours(method=cv2.CHAIN_APPROX_SIMPLE)

    edges = cc.image_edges[:, :, 0]

    corners = cv2.goodFeaturesToTrack(edges, 30, 0.03, 20)
    corners = np.int0(corners)

    img = cc.image_edges.copy()
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)


    cv2.imshow('corn', img)
    cv2.waitKey()

main()

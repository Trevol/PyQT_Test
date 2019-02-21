import cv2
from UI.contours_collector.detect_ellipses import detect_and_visualize
import utils
import time
import video_sources


def main():
    video = cv2.VideoCapture(video_sources.video_6)
    cv2.namedWindow('video', flags=cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO)

    while (1):
        ret, bgr = video.read()
        if not ret:
            break

        t0 = time.time()
        im = detect_and_visualize(bgr, visualize_edges=False)
        print(round(time.time() - t0, 3))

        cv2.imshow('video', im)
        if cv2.waitKey() == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

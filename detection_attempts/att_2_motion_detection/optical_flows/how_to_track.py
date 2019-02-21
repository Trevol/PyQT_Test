import cv2
import numpy as np
from cv_named_window import CvNamedWindow
from video_controller import VideoController
from video_capture import VideoCapture
import video_sources

def main():
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    video = VideoCapture(video_sources.video_2)
    wnd = CvNamedWindow('video')
    vc = VideoController(delay=50)

    prev_gray = None
    po = None
    tracking = False

    for frame in video.frames():
        if tracking:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
            p1 = p1[st == 1]
            # draw the tracks
            for pt in p1:
                a, b = pt.ravel()
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
            prev_gray = frame_gray
            p0 = p1.reshape(-1, 1, 2)

        wnd.imshow(frame)

        key = vc.wait_key()
        if key == 27:
            break
        elif not tracking and key == ord('r'):  # init tracking
            roi = cv2.selectROI('roi', frame)
            cv2.destroyWindow('roi')

            if roi is None or sum(roi) == 0:
                continue

            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=roi_mask(prev_gray, roi), **feature_params)
            if p0 is not None and len(p0) > 0:
                tracking = True

    video.release()


def roi_mask(base_im, roi):
    mask = np.zeros(base_im.shape[:2], np.uint8)
    x, y, w, h = roi
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


if __name__ == '__main__':
    main()

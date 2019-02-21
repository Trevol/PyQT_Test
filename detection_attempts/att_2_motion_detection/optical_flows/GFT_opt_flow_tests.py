import numpy as np
import cv2
from cv_named_window import CvNamedWindow
from video_controller import VideoController


def rect(image, pt, dim):
    (x, y), (w, h) = pt, dim
    image[y:y + h, x:x + w] = 255
    return image


def rect_frames_sequence(image_dim, rect_dim, from_pt, steps, step=1):
    from_x, from_y = from_pt
    for i in range(steps):
        image = np.zeros(image_dim, np.uint8)
        delta = i * step
        rect(image, (from_x + delta, from_y + delta), rect_dim)
        yield image


def ellipse_frames_sequence(image_dim, center0, axes, angle, steps, step=1):
    from_x, from_y = center0
    for i in range(steps):
        image = np.zeros(image_dim, np.uint8)
        delta = i * step
        cv2.ellipse(image, (from_x + delta, from_y + delta), axes, angle, 0, 360, 255, -1)
        yield image


def draw_points(image, pts, color=127):
    for pt in pts:
        x, y = pt.ravel().round(0).astype(np.int32)
        image[y, x] = color
    return image


def main():
    wnd = CvNamedWindow('frames', flags=cv2.WINDOW_NORMAL)
    vc = VideoController(delay=-1)

    # frames = rect_frames_sequence((200, 200), (40, 50), (5, 5), 50, step=2)
    frames = ellipse_frames_sequence((200, 200), (45, 55), (40, 50), 23, steps=30, step=2)
    frame0 = next(frames)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    pts0 = cv2.goodFeaturesToTrack(frame0, mask=None, **feature_params)
    # TODO: add point at center of rect
    # additional_pts = (
    #     [[5 + 40 / 2, 5 + 50 / 2]],
    #     [[4, 4]],
    #     [[3, 3]],
    #     [[5, 5]],
    #     [[2, 2]],
    #     [[0, 0]],
    #     [[-1, -1]],
    #     #-------------
    #     [[47, 57]],
    #     [[49, 59]],
    #     #-------------
    #     [[4, 6]],
    #     [[4, 8]],
    #     [[4, 9]],
    #     [[4, 11]],
    #     [[4, 12]],
    #     [[4, 13]],
    # )
    # pts0 = np.append(pts0, additional_pts, axis=0).astype(np.float32)

    if pts0 is None:
        print('No good features to track')
        return

    wnd.imshow(draw_points(frame0.copy(), pts0))
    vc.wait_key()

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    for frame in frames:
        pts1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame, pts0, None, **lk_params)
        pts1 = pts1[st == 1]

        wnd.imshow(draw_points(frame.copy(), pts1))
        pts0 = pts1.reshape(-1, 1, 2)
        frame0 = frame
        if vc.wait_key() == 27:
            break


if __name__ == '__main__':
    main()

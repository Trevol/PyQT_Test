"""
Region proposal logic by frames difference
"""
import cv2
import numpy as np
import utils

from detection_attempts.video_capture import VideoCapture


def get_frame(frame_pos):
    video = get_video()
    frame = video.read_at_pos(frame_pos)
    video.release()
    return frame


def get_video():
    source = 'd:/DiskE/Computer_Vision_Task/Video_6.mp4'
    return VideoCapture(source)


def main():
    video = get_video()
    prev = cv2.cvtColor(video.read(), cv2.COLOR_BGR2GRAY)
    for current in video.frames():
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(current, prev)
        cv2.imshow('diff', diff)

        thresh_binary_val, thresh_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh_binary', thresh_binary)
        thresh_otsu_val, thresh_otsu = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('thresh_otsu', thresh_otsu)
        print('diff', diff.min(), diff.max(), 'thresh_binary', thresh_binary.min(), thresh_binary.max(),
              thresh_binary_val, 'thresh_otsu', thresh_otsu.min(), thresh_otsu.max(), thresh_otsu_val)

        prev = current.copy()

        utils.put_frame_pos(current, video.frame_pos())
        cv2.imshow('current', current)

        if cv2.waitKey() == 27:
            break
    video.release()


def main_contacts_diff():
    video = get_video()
    prev = video.read_at_pos(115)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = video.read_at_pos(211)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev, next)
    cv2.imshow('diff', diff)

    diff_gray = cv2.absdiff(prev_gray, next_gray)
    cv2.imshow('diff_gray', diff_gray)

    def mouse_callback(evt, x, y, _, image):
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        print(evt, x, y, image[y, x], image[y - 5:y + 5, x - 5:x + 5].max())

    cv2.setMouseCallback('diff_gray', mouse_callback, param=diff_gray)

    thresh_otsu_val, thresh_otsu = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('thresh_otsu', thresh_otsu)

    thresh_binary_val, thresh_binary = cv2.threshold(diff_gray, 35, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh_binary', thresh_binary)

    cv2.waitKey()

    # 211 - dst
    # 115
    pass


# main_contacts_diff()
if __name__ == '__main__':
    main()
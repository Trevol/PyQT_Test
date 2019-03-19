import numpy as np
import cv2
from video_capture import VideoCapture
import utils
from cv_named_window import CvNamedWindow
from video_controller import VideoController
import video_sources

def denoise(frame):
    # frame = cv2.medianBlur(frame, 5, dst=frame)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0, dst=frame)
    frame = cv2.GaussianBlur(frame, (3, 3), 0, dst=frame)
    return frame


def main():
    video = VideoCapture(video_sources.video_6)

    diff_wnd = CvNamedWindow('diff')
    mask_wnd = CvNamedWindow('mask')
    input_wnd = CvNamedWindow('input')

    # prev_frame = denoise(video.read())
    prev_frame = cv2.cvtColor(denoise(video.read()), cv2.COLOR_BGR2GRAY)

    vm = VideoController(10)
    diff = None
    for frame in video.frames():
        # with utils.timeit_context('frame processing'):
        #     frame = denoise(frame)
        #     diff = cv2.absdiff(prev_frame, frame, dst=diff)
        #     ret, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        #     binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     # cn = cv2.countNonZero(binary_mask)
        #     nonzero = cv2.findNonZero(binary_mask)

        with utils.timeit_context('frame processing'):
            frame = denoise(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, gray, dst=diff)
            ret, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
            # cn = cv2.countNonZero(binary_mask)
            nonzero = cv2.findNonZero(mask)

        diff_wnd.imshow(diff)
        mask_wnd.imshow(mask)
        input_wnd.imshow(utils.put_frame_pos(frame.copy(), video.frame_pos()))

        prev_frame = gray

        if not vm.wait_key():
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

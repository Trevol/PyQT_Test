import numpy as np
import cv2
from cv_named_window import CvNamedWindow as Wnd
from video_capture import VideoCapture
from video_controller import VideoController
import video_sources
import utils
from image_resizer import resize


class WorkAreaView:
    def __init__(self, original_marker_points):
        self.original_marker_points = original_marker_points
        self.original_rect, rect_dims = self.bounding_rect(original_marker_points)
        self.marker_points = self.move_to_origin(original_marker_points)
        self.mask, self.mask_3ch = self.build_masks(self.marker_points, rect_dims)

    @staticmethod
    def build_masks(poly, dims):
        w, h = dims
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        mask_3ch = np.dstack((mask, mask, mask))
        return mask, mask_3ch

    @staticmethod
    def bounding_rect(poly):
        min_x = poly[..., 0].min()
        max_x = poly[..., 0].max() + 1
        min_y = poly[..., 1].min()
        max_y = poly[..., 1].max() + 1
        return (min_x, max_x, min_y, max_y), (max_x - min_x, max_y - min_y)

    @staticmethod
    def move_to_origin(poly):
        min_x = poly[..., 0].min()
        min_y = poly[..., 1].min()
        return np.dstack((poly[..., 0] - min_x, poly[..., 1] - min_y))

    def draw(self, image, color=(0, 255, 0), thickness=1):
        cv2.polylines(image, [self.marker_points], True, color, thickness)
        return image

    def mask_non_area(self, work_area_rect):
        mask = self.mask if len(work_area_rect.shape) == 2 else self.mask_3ch
        return cv2.bitwise_and(work_area_rect, mask, dst=work_area_rect)

    def extract_view(self, frame, denoise=True):
        x0, x1, y0, y1 = self.original_rect
        work_area_rect = frame[y0:y1, x0:x1]
        if denoise:
            work_area_rect = self.denoise(work_area_rect)
        work_area_rect = self.mask_non_area(work_area_rect)
        return work_area_rect

    @staticmethod
    def denoise(frame):
        return cv2.GaussianBlur(frame, (3, 3), 0, dst=frame)

    def skip_non_area(self, original_frames):
        for frame in original_frames:
            view = self.extract_view(frame)
            yield view, frame


class Helper:
    @staticmethod
    def exists_motion(frame, prev_frame, area_threshold=75):
        bin_diff, gray_diff = Helper.binary_diff(Helper.to_gray(frame), Helper.to_gray(prev_frame))
        cnt = cv2.countNonZero(bin_diff)
        if area_threshold < cnt < 1000:
            print('cnt', cnt)
        return cnt > area_threshold, bin_diff, gray_diff

    @staticmethod
    def binary_diff(gray1, gray2, thresh=50):
        gray_diff = cv2.absdiff(gray1, gray2)

        _, bin_diff = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
        return bin_diff, gray_diff

    @staticmethod
    def to_gray(bgr, dst=None):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY, dst=dst)

    @staticmethod
    def to_bgr(gray, dst=None):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR, dst=dst)

    @staticmethod
    def put_motion_state(img, motion_state, xy=(2, 2), wh=(20, 20)):
        gray = (127, 127, 127)
        red = (0, 0, 255)
        color = gray if motion_state == MotionState.NoMotion else red
        cv2.rectangle(img, xy, tuple(np.add(xy, wh)), color, -1)
        return img


def memoize_work_area_background(frames, video_controller, wnd):
    # TODO: отслеживать неск. кадров, т.к. отслеживание может начинаться с движения
    # find empty work area frame - this is last frame before motion detection
    wa_frame_prev, _ = next(frames)
    for wa_frame, _ in frames:
        if Helper.exists_motion(wa_frame, wa_frame_prev)[0]:
            return wa_frame_prev, wa_frame
        wa_frame_prev = wa_frame
        wnd.imshow(wa_frame)
        if video_controller.wait_key() == 27: break


class MotionState:
    NoMotion = 1
    InMotion = 2


def main():
    video = VideoCapture(video_sources.video_6)
    work_area = WorkAreaView(video_sources.video_6_work_area_markers)

    vc = VideoController(10, 'pause')
    (video_wnd, bin_diff_wnd, gray_diff_wnd, frame0_diff_wnd) = Wnd.create('video', 'binary diff', 'gray diff',
                                                                           'diff with frame0')

    frames_iter = work_area.skip_non_area(video.frames())

    frame0, prev_frame = memoize_work_area_background(frames_iter, vc, video_wnd)
    prev_state = MotionState.InMotion  # because empty background detection ends with motion

    for frame, _ in frames_iter:
        motion, bin_diff, gray_diff = Helper.exists_motion(frame, prev_frame)
        if motion:
            current_state = MotionState.InMotion
        else:
            current_state = MotionState.NoMotion

        # motion -> prevState = NoMotion and currentState == Motion
        motion_ended = prev_state == MotionState.InMotion and current_state == MotionState.NoMotion
        if motion_ended:
            frame0_diff = cv2.absdiff(frame0, frame)
            gray_of_color_diff = Helper.to_gray(frame0_diff)

            frame0_diff_wnd.imshow(resize(np.hstack((frame0_diff, Helper.to_bgr(gray_of_color_diff))), .5))

            val, binary = cv2.threshold(gray_of_color_diff, 35, 255, cv2.THRESH_BINARY)
            cv2.imshow('1 binary', resize(binary, .5))

            print('motion ended!', val)

        # VIS
        vis_img = Helper.put_motion_state(frame.copy(), current_state)
        vis_img = utils.put_frame_pos(vis_img, video.frame_pos(), xy=(2, 55))
        video_wnd.imshow(vis_img)
        bin_diff_wnd.imshow(resize(bin_diff, .5))
        gray_diff_wnd.imshow(resize(gray_diff, .5))
        # VIS END

        prev_frame, prev_state = frame, current_state
        if vc.wait_key() == 27: break

    video.release()


if __name__ == '__main__':
    main()

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
    def imdiff(img1, img2, area_threshold=75):
        bin_diff, gray_diff = Helper.binary_diff(Helper.to_gray(img1), Helper.to_gray(img2))
        cnt = cv2.countNonZero(bin_diff)
        differs = cnt > area_threshold
        return differs, bin_diff, gray_diff

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


class MotionDetector:
    class MotionState:
        Silence = 1
        Motion = 2
        EnsuringSilence = 3

    def __init__(self, initial_frame, frames_num):
        self.frames_num = frames_num
        self.current_frame = initial_frame
        self.current_state = self.MotionState.Silence
        self.bin_diff = self.gray_diff = None
        self.prev_frame = initial_frame
        self.prev_state = self.MotionState.Silence
        self.ensuringSilenceCounter = None

    def put_current_state(self, frame, xy=(2, 2), wh=(20, 20)):
        gray = (127, 127, 127)
        red = (0, 0, 255)
        color = gray if self.current_state == self.MotionState.Silence else red
        cv2.rectangle(frame, xy, tuple(np.add(xy, wh)), color, -1)
        return frame

    # def detect(self, frame):
    #     if self.prev_frame is None:
    #         raise Exception('prev_frame is None')
    #     self.prev_state = self.current_state
    #     self.prev_frame = self.current_frame
    #
    #     frame_differ_from_prev, self.bin_diff, self.gray_diff = Helper.imdiff(frame, self.prev_frame)
    #
    #     self.current_state = self.MotionState.Motion if frame_differ_from_prev else self.MotionState.Silence
    #     self.current_frame = frame

    def detect(self, currentFrame):
        if self.prev_frame is None:
            raise Exception('prev_frame is None')
        self.prev_state = self.current_state
        self.prev_frame = self.current_frame
        self.current_frame = currentFrame
        motionDetected, self.bin_diff, self.gray_diff = Helper.imdiff(self.current_frame, self.prev_frame)

        State = self.MotionState
        if self.prev_state == State.Silence:
            if motionDetected:
                self.current_state = State.Motion
            else:
                self.current_state = State.Silence

        elif self.prev_state == State.Motion:
            if motionDetected:
                self.current_state = State.Motion
            else:
                self.current_state = State.EnsuringSilence
                self.ensuringSilenceCounter = 1

        elif self.prev_state == State.EnsuringSilence:
            if motionDetected:
                self.current_state = State.Motion
                self.ensuringSilenceCounter = None
            else:
                if self.ensuringSilenceCounter >= self.frames_num:
                    self.current_state = State.Silence
                    self.ensuringSilenceCounter = None
                else:
                    self.current_state = State.EnsuringSilence
                    self.ensuringSilenceCounter = self.ensuringSilenceCounter + 1

        else:
            raise Exception(f'Unexpected state {self.prev_state}')

    def motion_ended(self):
        return self.prev_state != self.MotionState.Silence and self.current_state == self.MotionState.Silence

    def motion_started(self):
        return self.prev_state != self.MotionState.Motion and self.current_state == self.MotionState.Motion


class BackgroundModel:
    def __init__(self, motion_detector, framesHistoryLen):
        self.motion_detector = motion_detector
        self.framesHistoryLen = framesHistoryLen
        self.__accumulated = np.zeros(motion_detector.prev_frame.shape, np.float32)
        self.frames_learned = 0
        self.learned = None
        self.done = False

    @staticmethod
    def __average(accumulatedSamples, samplesCount):
        accumulatedSamples = np.divide(accumulatedSamples, samplesCount, out=accumulatedSamples)
        accumulatedSamples = np.round(accumulatedSamples, 0, out=accumulatedSamples)
        return accumulatedSamples.astype(np.uint8)

    def learn(self):
        if self.done:
            raise Exception('already done!')

        cv2.accumulate(self.motion_detector.prev_frame, self.__accumulated)
        self.frames_learned = self.frames_learned + 1

        if self.motion_detector.motion_started():
            self.learned = self.__average(self.__accumulated, self.frames_learned)
            self.done = True
            self.__accumulated = None

        return self.done


def main():
    video = VideoCapture(video_sources.video_6)
    work_area = WorkAreaView(video_sources.video_6_work_area_markers)

    vc = VideoController(10, 'pause')
    (video_wnd, bin_diff_wnd,
     gray_diff_wnd, frame0_diff_wnd,
     learned_BG_wnd) = Wnd.create('video', 'binary diff', 'gray diff', 'diff with frame0', 'Learned BG')

    frames_iter = work_area.skip_non_area(video.frames())

    motion_detector = MotionDetector(next(frames_iter)[0], 3)
    background = BackgroundModel(motion_detector, 15)

    for frame, _ in frames_iter:
        motion_detector.detect(frame)
        if not background.done:
            background.learn()
        else:
            if motion_detector.motion_ended():
                frame0_diff = cv2.absdiff(background.learned, frame)
                gray_of_color_diff = Helper.to_gray(frame0_diff)

                frame0_diff_wnd.imshow(resize(np.hstack((frame0_diff, Helper.to_bgr(gray_of_color_diff))), .5))

                _, binary = cv2.threshold(gray_of_color_diff, 35, 255, cv2.THRESH_BINARY)
                cv2.imshow('1 binary', resize(binary, .5))

                # VIS
        if background.done:
            learned_BG_wnd.imshow(resize(background.learned, 1))

        vis_img = motion_detector.put_current_state(frame.copy())
        vis_img = utils.put_frame_pos(vis_img, video.frame_pos(), xy=(2, 55))
        video_wnd.imshow(vis_img)
        # bin_diff_wnd.imshow(resize(motion_detector.bin_diff, .5))
        # gray_diff_wnd.imshow(resize(motion_detector.gray_diff, .5))
        # VIS END

        if vc.wait_key() == 27: break

    video.release()


if __name__ == '__main__':
    main()

import numpy as np
import cv2
from cv_named_window import CvNamedWindow as Wnd
from video_capture import VideoCapture
from video_controller import VideoController
import video_sources
import utils
from image_resizer import resize, ImageResizer
from collections import deque


class WorkAreaView:
    def __init__(self, original_marker_points):
        self.original_marker_points = original_marker_points
        self.original_rect, self.rect_dims = self.bounding_rect(original_marker_points)
        self.marker_points = self.move_to_origin(original_marker_points)
        self.mask, self.mask_3ch = self.build_masks(self.marker_points, self.rect_dims)

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

    def resolution(self):
        return self.rect_dims

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
            work_area_rect = self.__denoise(work_area_rect)
        work_area_rect = self.mask_non_area(work_area_rect)
        return work_area_rect

    @staticmethod
    def __denoise(frame, dst=None):
        return cv2.GaussianBlur(frame, (3, 3), 0, dst=dst)

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

    def indicateCurrentState(self, frame, xy=(2, 2), wh=(20, 20)):
        gray = (127, 127, 127)
        red = (0, 0, 255)
        color = gray if self.current_state == self.MotionState.Silence else red
        cv2.rectangle(frame, xy, tuple(np.add(xy, wh)), color, -1)
        return frame

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

    def motionEnded(self):
        return self.prev_state != self.MotionState.Silence and self.current_state == self.MotionState.Silence

    def motionStarted(self):
        return self.prev_state == self.MotionState.Silence and self.current_state == self.MotionState.Motion

    def isSilence(self):
        return self.current_state == self.MotionState.Silence

    def isMotion(self):
        return self.current_state == self.MotionState.Motion


class BackgroundModel:
    def __init__(self, framesHistoryLen):
        if framesHistoryLen < 1:
            raise Exception('framesHistoryLen < 1')
        self.__framesHistoryAccumulator = None
        self.__framesHistory = deque(maxlen=framesHistoryLen)
        self.learned = None

    def learn(self, frame, foregroundMask=None):
        if self.__framesHistoryAccumulator is None:
            self.__framesHistoryAccumulator = np.zeros(frame.shape, np.float32)
            self.learned = np.zeros(frame.shape, np.uint8)
        history = self.__framesHistory
        accumulator = self.__framesHistoryAccumulator

        if len(history) == history.maxlen:  # if frames queue is full
            oldestFrame, bgMask = history.popleft()
            cv2.subtract(accumulator, oldestFrame, dst=accumulator, dtype=cv2.CV_32F,
                         mask=bgMask)  # remove old frame from accum

        bgMask = cv2.bitwise_not(foregroundMask)
        history.append((frame, bgMask))
        cv2.accumulate(frame, accumulator, mask=bgMask)  # accumulate frames sum
        self.learned = np.true_divide(accumulator, len(history), out=self.learned, casting='unsafe')  # average frame


class Segmenter:
    def __init__(self):
        self._labels = None
        self._mask = None
        self._distTransform = None
        self._thresh = None
        self._sureFg = None
        self._unknown = None
        self._bgr = None

    def segment(self, binaryImage):
        # fill holes
        if self._labels is None:
            self._labels = np.empty(binaryImage.shape, np.int32)
        cnt, labels = cv2.connectedComponents(binaryImage, ltype=cv2.CV_32S, labels=self._labels)

        cv2.floodFill(labels, None, (0, 0), cnt)  # fill main background
        mask = np.equal(labels, 0, out=self._mask)
        binaryImage[mask] = 255

        # prepare sure FG:
        distTransform = cv2.distanceTransform(binaryImage, cv2.DIST_L1, 3, dst=self._distTransform, dstType=cv2.CV_8U)
        _, sureFg = cv2.threshold(distTransform, 0.8 * distTransform.max(), 255, cv2.THRESH_BINARY, dst=self._thresh)

        unknown = cv2.subtract(binaryImage, sureFg, dst=self._unknown)

        cnt, markers = cv2.connectedComponents(sureFg, labels=self._labels)

        cv2.add(markers, 1, dst=markers)  # background should be 1

        mask = np.equal(unknown, 255, out=self._mask)
        markers[mask] = 0  # label unknown regions with 0

        bgrImg = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR, dst=self._bgr)
        markers = cv2.watershed(bgrImg, markers)
        return markers, cnt - 1

    @staticmethod
    def markersToDisplayImage(markers, objectsCount):
        displayImage = np.zeros([*markers.shape[:2], 3], np.uint8)

        for objectLabel in range(objectsCount):
            objectLabel = objectLabel + 2
            rr, cc = np.where(markers == objectLabel)
            displayImage[rr, cc] = utils.random_color()

        return displayImage


def main():
    video = VideoCapture(video_sources.video_2)
    workArea = WorkAreaView(video_sources.video_2_work_area_markers)

    vc = VideoController(10, 'pause')
    (video_wnd, bin_diff_wnd,
     gray_diff_wnd, colorDiffWnd,
     learned_BG_wnd) = Wnd.create('video', 'binary diff', 'gray diff', 'color diff', 'Learned BG')
    colorAbsDiffWnd = Wnd('color_abs_diff')
    segmentedWnd = Wnd('segmented')

    segmenter = Segmenter()

    frames_iter = workArea.skip_non_area(video.frames())

    motionDetector = MotionDetector(next(frames_iter)[0], 3)
    backgroundModel = BackgroundModel(15)
    prevBackground = None
    for frame, _ in frames_iter:
        motionDetector.detect(frame)

        if motionDetector.motionEnded():
            # calc fgMask
            mask, gray_diff, color_diff, colorAbsDiff = calcForegroundMask(prevBackground, frame)
            bin_diff_wnd.imshow(resize(mask, .5))
            # gray_diff_wnd.imshow(resize(gray_diff, .5))
            # colorDiffWnd.imshow(resize(color_diff, .5))
            # colorAbsDiffWnd.imshow(resize(colorAbsDiff, .5))
            markers, objectsCount = segmenter.segment(mask)
            segmentedWnd.imshow(resize(Segmenter.markersToDisplayImage(markers, objectsCount), .5))

            backgroundModel = BackgroundModel(15)

        if motionDetector.isSilence():
            backgroundModel.learn(frame, foregroundMask=None)
            learned_BG_wnd.imshow(resize(backgroundModel.learned, .5))

        if motionDetector.motionStarted():
            prevBackground = backgroundModel.learned
            backgroundModel = None
            learned_BG_wnd.imshow(resize(prevBackground, .5))

        # VIS

        vis_img = motionDetector.indicateCurrentState(frame.copy())
        vis_img = utils.put_frame_pos(vis_img, video.frame_pos(), xy=(2, 55))
        video_wnd.imshow(vis_img)
        # bin_diff_wnd.imshow(resize(motionDetector.bin_diff, .5))
        # gray_diff_wnd.imshow(resize(motionDetector.gray_diff, .5))
        # VIS END

        if vc.wait_key() == 27: break

    video.release()


thresh = 10


def calcForegroundMask_abs(bg, newStateFrame):
    color_diff = cv2.absdiff(bg, newStateFrame)
    gray_diff = Helper.to_gray(color_diff)
    _, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
    return mask, gray_diff, color_diff


def calcForegroundMask(bg, newStateFrame):
    color_diff = cv2.subtract(newStateFrame, bg)
    gray_diff = Helper.to_gray(color_diff)
    _, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)

    color_abs_diff = cv2.absdiff(newStateFrame, bg)

    return mask, gray_diff, color_diff, color_abs_diff


if __name__ == '__main__':
    main()

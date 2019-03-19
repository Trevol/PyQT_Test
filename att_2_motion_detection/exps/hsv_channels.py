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


def main():
    video = VideoCapture(video_sources.video_6)
    work_area = WorkAreaView(video_sources.video_6_work_area_markers)

    vc = VideoController(10, 'pause')
    video_wnd, = Wnd.create('video')
    # h_wnd, s_wnd, v_wnd = Wnd.create('H', 'S', 'V')
    # L_wnd, a_wnd, b_wnd = Wnd.create('L', 'a', 'b')

    frames_iter = work_area.skip_non_area(video.frames())
    for frame, _ in frames_iter:
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # h_wnd.imshow(hsv[:, :, 0])
        # s_wnd.imshow(hsv[:, :, 1])
        # v_wnd.imshow(hsv[:, :, 2])

        # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        # L_wnd.imshow(lab[:, :, 0])
        # a_wnd.imshow(lab[:, :, 1])
        # b_wnd.imshow(lab[:, :, 2])

        vis_img = utils.put_frame_pos(frame, video.frame_pos(), xy=(2, 55))
        video_wnd.imshow(vis_img)

        if vc.wait_key() == 27: break

    video.release()


if __name__ == '__main__':
    main()

import cv2


class VideoCapture:
    cap = None

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)

    def frames(self):
        while 1:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def frame_pos(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def frame_pos_msec(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def read(self):
        _, frame = self.cap.read()
        return frame

    def set_pos(self, frame_pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    def read_at_pos(self, pos):
        self.set_pos(pos)
        return self.read()

    def resolution(self):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_width, frame_height

    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    # def fourcc(self):
    #     return int(self.cap.get(cv2.CAP_PROP_FOURCC))

    def release(self):
        self.cap.release()

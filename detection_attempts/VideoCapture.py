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

    def read(self):
        _, frame = self.cap.read()
        return frame

    def set_pos(self, frame_pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    def read_at_pos(self, pos):
        self.set_pos(pos)
        return self.read()

    def release(self):
        self.cap.release()

from atom.api import Atom, Typed, Int
import cv2


class VideoModel(Atom, object):
    video = Typed(cv2.VideoCapture)
    current_pos = Int(1)

    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)

    def frame_interval(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return int(1000 // fps)

    def frames_count(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def next_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        self.current_pos = self.current_pos + 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def destroy(self):
        self.video.release()

from atom.api import Atom, Typed, Int, observe
from enaml.widgets.api import Timer
import cv2
import numpy as np

from enaml.widgets.api import Slider

class _VideoModel:
    _timer = None
    _in_timer_proc = False

class VideoModel(Atom, _VideoModel):
    video = Typed(cv2.VideoCapture)
    current_frame_pos = Int(1)
    current_frame = Typed(np.ndarray)

    def __init__(self, video_source, timer):
        self.video = cv2.VideoCapture(video_source)
        self._timer = timer
        self._timer.interval = self._frame_interval()
        self._timer.observe('timeout', self._next_frame)

    @observe('current_frame_pos')
    def current_frame_pos_change(self, change):
        if self._in_timer_proc:
            return
        # pos changed by outer world
        self._timer.stop()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
        self._timer.start()

    def start(self):
        self._timer.start()

    def _frame_interval(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return int(1000 // fps)

    def frames_count(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def _next_frame(self, _):
        try:
            self._in_timer_proc = True
            ret, frame = self.video.read()
            if not ret:
                self._timer.stop()
                return
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_pos = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))  # self.current_frame_pos + 1
            self._put_current_info()
        finally:
            self._in_timer_proc = False

    def _put_current_info(self):
        cv2.putText(self.current_frame, str(self.current_frame_pos), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)

    def destroy(self):
        self.video.release()

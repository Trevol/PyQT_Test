import cv2
from enaml.widgets.api import ImageView
from enaml.image import Image
from enaml.core.declarative import d_
from atom.api import Typed, Bool
import numpy as np


class NpImageView(ImageView):
    npimage = d_(Typed(np.ndarray))
    convert_RGB2BGR = d_(Bool(True))

    def __init__(self, parent):
        super(NpImageView, self).__init__(parent)
        self.npimage # read to init observer

    def _observe_npimage(self, change):
        if self.npimage is None:
            self.npimage = None
        else:
            if len(self.npimage.shape) == 3 and self.convert_RGB2BGR:
                np_image = cv2.cvtColor(self.npimage, cv2.COLOR_RGB2BGR)
            else:
                np_image = self.npimage
            _, imencoded = cv2.imencode('.bmp', np_image)
            self.image = Image(data=imencoded.tobytes())

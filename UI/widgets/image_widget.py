import numpy as np
from atom.api import (Typed, set_default, observe)
from enaml.core.declarative import d_
from enaml.widgets.api import RawWidget
import pyqtgraph as pg

import json

pg.setConfigOptions(imageAxisOrder='row-major')


class PgImageWidget(pg.GraphicsView):
    def __init__(self, parent, widget):
        super(PgImageWidget, self).__init__(parent, background=(0, 0, 0, 0))  # transparent background
        self.widget = widget

    def mouseReleaseEvent(self, ev):
        super(PgImageWidget, self).mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        # PyQt5.QtGui.QWheelEvent
        super(PgImageWidget, self).wheelEvent(ev)
        # print('wheelEvent', ev, ev.x(), ev.y(), ev.pos(), ev.globalPos(), ev.angleDelta(), ev.phase())

    def mousePressEvent(self, ev):
        super(PgImageWidget, self).mousePressEvent(ev)
        # print('mousePressEvent', ev)


class ImageWidget(RawWidget):
    hug_width = set_default('ignore')
    hug_height = set_default('ignore')

    image = d_(Typed(np.ndarray))
    image_item = Typed(pg.ImageItem)

    def create_widget(self, parent):
        widget = PgImageWidget(parent, self)
        self.image_item = pg.ImageItem()
        widget.addItem(self.image_item)
        self.image  # note: need read image to trigger observers
        return widget

    @observe('image')
    def on_image_update(self, change):
        # if change['name'] != 'image': return
        image = change['value']
        if image is None:
            self.image_item.clear()
        else:
            self.image_item.setImage(image)

    def update(self):
        self.proxy.widget and self.proxy.widget.update()

from enaml.widgets.api import ImageView
from enaml.widgets.image_view import ProxyImageView
from enaml.qt.qt_image_view import QtImageView, QImageView
from enaml.qt.qt_factories import QT_FACTORIES
from enaml.qt.QtGui import QImage, QPixmap
from atom.api import Typed
from enaml.core.declarative import d_func, d_
import numpy as np


# TODO: implement https://www.swharden.com/wp/2013-06-03-realtime-image-pixelmap-from-numpy-array-data-in-qt/
class QImageViewEx(QImageView):
    def mouseDoubleClickEvent(self, event):
        self.proxy.mouse_double_click_event(event)
        super(QImageViewEx, self).mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        self.proxy.wheel_event(event)
        super(QImageViewEx, self).wheelEvent(event)

    def mousePressEvent(self, event):
        self.proxy.mouse_press_event(event)
        super(QImageViewEx, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.proxy.mouse_release_event(event)
        super(QImageViewEx, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        self.proxy.mouse_move_event(event)
        super(QImageViewEx, self).mouseMoveEvent(event)

    def __init__(self, proxy, parent=None):
        super(QImageViewEx, self).__init__(parent)
        self.proxy = proxy


class QtImageViewEx(QtImageView):
    widget = Typed(QImageViewEx)

    def create_widget(self):
        self.widget = QImageViewEx(self, self.parent_widget())

    def set_image(self, image):
        """ Set the image on the underlying widget.
        """
        with self.geometry_guard():
            self.widget.setPixmap(_get_pixmap(image))


    def mouse_double_click_event(self, event):
        self.declaration.mouse_double_click_event(event)

    def wheel_event(self, event):
        self.declaration.wheel_event(event)

    def mouse_press_event(self, event):
        self.declaration.mouse_press_event(event)

    def mouse_release_event(self, event):
        self.declaration.mouse_release_event(event)

    def mouse_move_event(self, event):
        self.declaration.mouse_move_event(event)


class ImageViewEx(ImageView):
    image = d_(Typed(np.ndarray))

    @d_func
    def mouse_double_click_event(self, event): pass

    @d_func
    def wheel_event(self, event): pass

    @d_func
    def mouse_press_event(self, event): pass

    @d_func
    def mouse_release_event(self, event): pass

    @d_func
    def mouse_move_event(self, event): pass


def image_view_ex_factory():
    return QtImageViewEx


QT_FACTORIES.update({
    'ImageViewEx': image_view_ex_factory
})


def _get_pixmap(image):
    if image is None:
        return None
    w, h, format = _get_image_size_n_format(image)
    qimage = QImage(image.data, w, h, format)
    return QPixmap.fromImage(qimage)


def _get_image_size_n_format(image):
    if image.dtype != np.uint8:
        raise NotImplementedError(f"Unsupported image dtype {image.dtype}")
    h, w = image.shape[0:2]
    if len(image.shape) == 2:
        format = QImage.Format_Grayscale8
    else:
        format = QImage.Format_RGB888
    return w, h, format

def ndarray_to_qimage(arr):
    """
    Convert NumPy array to QImage object
    credits: https://github.com/PierreRaybaut/PythonQwt/blob/master/qwt/toqimage.py

    :param numpy.array arr: NumPy array
    :return: QImage object
    """
    # https://gist.githubusercontent.com/smex/5287589/raw/toQImage.py
    if arr is None:
        return QImage()
    if len(arr.shape) not in (2, 3):
        raise NotImplementedError("Unsupported array shape %r" % arr.shape)
    data = arr.data
    ny, nx = arr.shape[:2]
    stride = arr.strides[0]  # bytes per line
    color_dim = None
    if len(arr.shape) == 3:
        color_dim = arr.shape[2]
    if arr.dtype == np.uint8:
        if color_dim is None:
            qimage = QImage(data, nx, ny, stride, QImage.Format_Indexed8)
            #            qimage.setColorTable([qRgb(i, i, i) for i in range(256)])
            qimage.setColorCount(256)
        elif color_dim == 3:
            qimage = QImage(data, nx, ny, stride, QImage.Format_RGB888)
        elif color_dim == 4:
            qimage = QImage(data, nx, ny, stride, QImage.Format_ARGB32)
        else:
            raise TypeError("Invalid third axis dimension (%r)" % color_dim)
    elif arr.dtype == np.uint32:
        qimage = QImage(data, nx, ny, stride, QImage.Format_ARGB32)
    else:
        raise NotImplementedError("Unsupported array data type %r" % arr.dtype)
    return qimage


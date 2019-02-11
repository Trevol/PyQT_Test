import cv2
import time


class CvNamedWindow:
    def __init__(self, winname=None, flags=1, mouse_callback=None):
        self.winname = winname or f'unnamed-{time.time()}'
        self.current_img = None
        self.mouse_callback = mouse_callback
        cv2.namedWindow(self.winname, flags=flags)
        cv2.setMouseCallback(self.winname, self.__mouse_callback)

    def imshow(self, img):
        self.current_img = img
        cv2.imshow(self.winname, img)

    def __mouse_callback(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_RBUTTONDOWN:
            color = self.current_img[y, x] if self.current_img is not None else None
            print(f'Window "{self.winname}" flags: {flags}', (x, y), f'Color {color}')
        if self.mouse_callback is not None:
            self.mouse_callback(evt, x, y, flags, self.current_img)

    def destroy(self):
        cv2.destroyWindow(self.winname)

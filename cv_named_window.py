import os
import cv2
import time


# TODO: select ROI
# TODO: save ROI to file

class CvNamedWindow:
    class ROI:
        def __init__(self, wnd, roiSelectedEvent):
            self.__wnd = wnd
            self.__roi = None
            self.__roiSelectedEvent = roiSelectedEvent

        @staticmethod
        def __isEmpty(roi):
            x, y, w, h = roi or (0, 0, 0, 0)
            return x == 0 and y == 0 and w == 0 and h == 0

        def clear(self, redraw=True):
            self.__roi = None
            if redraw:
                self.__wnd.redraw()

        def select(self):
            if self.__wnd.current_img is None:
                return
            self.__wnd.setTitleStatus('ROI selection')
            roi = cv2.selectROI(self.__wnd.winname, self.__wnd.current_img)
            self.__roi = None if self.__isEmpty(roi) else roi

            if self.__roiSelectedEvent is not None:
                self.__roiSelectedEvent(self.__roi, self._imageRoi(self.__roi, self.__wnd.current_img), self.__wnd, self.__wnd.current_img)

            self.draw()

            self.__wnd._reinitMouseCallback()  # restore callback
            self.__wnd.setTitleStatus(None)  # restore title

        @staticmethod
        def _imageRoi(roi, img):
            if roi is None:
                return None
            x, y, w, h = roi
            return img[y:y + h + 1, x:x + w + 1]

        def draw(self):
            if self.__isEmpty(self.__roi) or self.__wnd.current_img is None:
                self.__wnd.redraw()
                return
            x, y, w, h = self.__roi
            img = self.__wnd.current_img.copy()
            blue = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), blue, 1)
            self.__wnd.redraw(img)

        def save(self):
            if self.__isEmpty(self.__roi) or self.__wnd.current_img is None:
                return
            x, y, w, h = self.__roi
            file = os.path.join(os.getcwd(), f'{x}_{y}_{w}_{h}_{time.time()}.png')
            roi = self.__wnd.current_img[y:y + h, x:x + w]
            cv2.imwrite(file, roi)
            print('ROI saved to', file)

    @classmethod
    def create(cls, *windowsArgs, **windowsKvArgs):
        windows = []
        for windowArg in windowsArgs:
            if isinstance(windowArg, dict):
                windows.extend(cls._createWindowsFromDict(windowArg))
            else:
                windows.append(cls(str(windowArg), None))
        windows.extend(cls._createWindowsFromDict(windowsKvArgs))
        return windows

    @classmethod
    def _createWindowsFromDict(cls, windowsKvArgs):
        for name, flags in windowsKvArgs.items():
            yield cls(name, flags)

    def __init__(self, winname=None, flags=1, mouse_callback=None, roiSelectedEvent=None):
        defaultFlags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED

        self.winname = winname or f'unnamed-{time.time()}'
        self.win_flags = flags if flags is not None else defaultFlags
        self.current_img = None
        self.roi = self.ROI(self, roiSelectedEvent)
        self.mouse_callback = mouse_callback
        self.window_created = False

    def imshow(self, img):
        self.__ensureWndCreated()
        self.current_img = img
        self.roi.clear(redraw=False)
        self.redraw()

    def redraw(self, img=None):
        if img is None:
            img = self.current_img
        if img is not None:
            cv2.imshow(self.winname, img)

    def __ensureWndCreated(self):
        if not self.window_created:
            cv2.namedWindow(self.winname, flags=self.win_flags)
            self._reinitMouseCallback()
            self.window_created = True

    def __mouse_callback(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_RBUTTONDOWN:
            color = self.current_img[y, x] if self.current_img is not None else None
            print(f'Window "{self.winname}" flags: {flags}', (x, y), f'Color {color}')
        elif evt == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.roi.select()
            elif flags & cv2.EVENT_FLAG_ALTKEY:
                self.roi.save()
            else:
                self.roi.clear(redraw=True)

        if self.mouse_callback is not None:
            self.mouse_callback(evt, x, y, flags, self.current_img)

    def setTitleStatus(self, status):
        title = f'{self.winname} ({status})' if status else self.winname
        cv2.setWindowTitle(self.winname, title)

    def _reinitMouseCallback(self):
        cv2.setMouseCallback(self.winname, self.__mouse_callback)

    def destroy(self):
        cv2.destroyWindow(self.winname)


def imshow(**windowImages):
    wnds = []
    for winname, img in windowImages.items():
        wnd = CvNamedWindow(winname)
        wnd.imshow(img)
        wnds.append(wnd)
    return wnds


def cvWaitKeys(*keys):
    if len(keys) == 0:  # wait any key
        return cv2.waitKey()
    keys = [_keyCode(k) for k in keys]
    while True:
        key = cv2.waitKey()
        if key == -1 or key in keys:
            return key


def _keyCode(key):
    return ord(key) if isinstance(key, str) else key

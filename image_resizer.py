import cv2


class ImageResizer:
    def __init__(self, src_resolution, scale=None, width=None, height=None):
        self.dst_resolution = self.compute_dst_resolution(src_resolution, scale, width, height)

    @staticmethod
    def compute_dst_resolution(src_resolution, scale=None, width=None, height=None):
        src_w, src_h = src_resolution

        if scale is not None:
            if scale == 1:
                return None # don't need to resize
            return int(round(src_w * scale)), int(round(src_h * scale))

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return None

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            scale = height / src_h
            return int(round(src_w * scale)), height

        # otherwise, the height is None
        # calculate the ratio of the width and construct the
        # dimensions
        scale = width / src_w
        return width, int(round(src_h * scale))

    def resize(self, image, dst=None):
        if self.dst_resolution is None:
            return image
        return cv2.resize(image, self.dst_resolution, interpolation=cv2.INTER_AREA, dst=dst)


def resize(image, scale=None, width=None, height=None):
    resolution = image.shape[1::-1] # flip height and width to create expected resolution (w, h)
    r = ImageResizer(resolution, scale, width, height)
    return r.resize(image)

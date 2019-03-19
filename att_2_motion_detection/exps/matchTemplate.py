import numpy as np
import cv2
import utils


def main():
    # shape = (1080, 1920, 3)
    shape = (165, 455, 3)
    image = np.zeros(shape, np.uint8)
    tpl_shape = (94, 94, 3)
    template = np.full(tpl_shape, 127, np.uint8)
    image[20:20 + template.shape[0], 20:20 + template.shape[1]] = template #+ np.random.random(tpl_shape)*75

    tmMethods = {cv2.TM_SQDIFF: 'TM_SQDIFF', cv2.TM_SQDIFF_NORMED: 'TM_SQDIFF_NORMED',
                 cv2.TM_CCORR: 'TM_CCORR', cv2.TM_CCORR_NORMED: 'TM_CCORR_NORMED',
                 cv2.TM_CCOEFF: 'TM_CCOEFF', cv2.TM_CCOEFF_NORMED: 'TM_CCOEFF_NORMED'}

    for method, methodName in tmMethods.items():
        r = cv2.matchTemplate(image, template, method)
        print(methodName, r.shape, r.dtype, r.min(), r.max())


if __name__ == '__main__':
    main()

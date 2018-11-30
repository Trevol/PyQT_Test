import utils
import numpy as np, cv2, time, math
from utils import intt, box_to_ellipse


def main():
    # t0 = time.time()
    # for i in range(10000):
    #     poly = utils.ellipseF2PolyRounded((67.999, 112.999), (22.999, 33.999), 16.999, 0, 360, 1)
    # print(round(time.time()-t0, 3))

    # t0 = time.time()
    # for i in range(10000):
    #     poly = utils.ellipseF2Poly((67.999, 112.999), (22.999, 33.999), 16.999, 0, 360, 1)
    # print(round(time.time() - t0, 3))

    # t0 = time.time()
    # for i in range(10000):
    #     poly = utils.ellipseF2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    # print(round(time.time() - t0, 3))

    t0 = time.time()
    for i in range(100000):
        poly = utils.ellipseF2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    print(round(time.time() - t0, 3))

    t0 = time.time()
    for i in range(100000):
        poly = utils.ellipseF2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    print(round(time.time() - t0, 3))

    t0 = time.time()
    for i in range(10000):
        poly = utils.ellipseF2PolyRounded((68, 113), (23, 34), 17, 0, 360, 1)
    print(round(time.time() - t0, 3))

    t0 = time.time()
    for i in range(10000):
        poly = utils.ellipseF2PolyRounded((68, 113), (23, 34), 17, 0, 360, 1)
    print(round(time.time() - t0, 3))

    t0 = time.time()
    for i in range(10000):
        poly = cv2.ellipse2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    print(round(time.time() - t0, 3))



def test():
    poly = utils.ellipseF2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    print(poly.shape, poly.dtype)

    poly2 = utils.ellipseF2Poly_stack((68, 113), (23, 34), 17, 0, 360, 1)
    print(poly2.shape, poly2.dtype, np.array_equal(poly, poly2))


    poly = utils.ellipseF2PolyRounded((68, 113), (23, 34), 17, 0, 360, 1)
    print(poly.shape, poly.dtype)

    poly = cv2.ellipse2Poly((68, 113), (23, 34), 17, 0, 360, 1)
    print(poly.shape, poly.dtype)

def test_distances():
    img_ellipse = np.zeros((300, 300), dtype=np.uint8)

    center, axes, angle = (150.6, 150.34), (100.66, 50.77), 16.58

    cv2.ellipse(img_ellipse, intt(center), intt(axes), angle, 0, 360, 255, thickness=2)
    contour = cv2.findContours(img_ellipse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0]

    fitted_ellipse = box_to_ellipse(*cv2.fitEllipseDirect(contour))

    contour = utils.normalize_contour(contour)



    poly_cv = cv2.ellipse2Poly(intt(center), intt(axes), round(angle), 0, 360, 1)
    print(utils.polygon_polygon_test(poly_cv, contour, -1))
    print(utils.polygon_polygon_test(contour, poly_cv, -1))

    poly_my = utils.ellipseF2Poly_numpy(center, axes, angle, 0, 360, 1)
    poly_my = np.round(poly_my, 0, out=poly_my).astype(np.int32)
    print(utils.polygon_polygon_test(poly_my, contour, -1))
    print(utils.polygon_polygon_test(contour, poly_my, -1))


main()

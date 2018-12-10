import cv2
import numpy as np
from detection_attempts.att_1.contour import Contour
from contour_visualize import draw_contours
import geometry_utils as geometry
import utils
from detection_attempts.att_1.detector import Detector
from detection_attempts.VideoCapture import VideoCapture

def test_image():
    im = np.zeros((300, 300), dtype=np.uint8)

    cv2.ellipse(im, (100, 100), (50, 30), 0, 180, 360, 255, 1)
    cv2.ellipse(im, (200, 100), (50, 30), 0, 180, 360, 255, 1)

    # cv2.ellipse(im, (200, 100), (50, 30), 16, 0, 360, 255, 1)

    return im


def split_1(contour):
    # сразу сравнивать с эталонным элипсом???
    pts = contour.measurements().approx_points
    angles, pts_unwrapped = geometry.compute_angles_vectorized(pts)

    # (45. < angles) & (angles <= 160.)
    # mask = angles > 160
    mask = 45 < angles
    indexes = np.where(mask)[0]
    if indexes.size == 0:
        return np.empty((0, 2)), np.empty((0, 1, 2))
    tails = pts_unwrapped[indexes]

    parts_num = len(indexes)
    parts = [None] * parts_num

    for i in range(parts_num - 1):
        index = indexes[i]
        next_index = indexes[i + 1]
        parts[i] = pts[index: next_index + 1]

    parts[-1] = np.append(pts[indexes[-1]:], pts[:indexes[0] + 1], axis=0)

    ############################
    unique_parts = [parts[0]]
    for i in range(1, parts_num):
        p_i = parts[i]

        # если отличается ото всех уникальных - добавляем к уникальным
        # если найден подобный контур -
        if Detector.first_or_default(unique_parts, p_i, polygons_equiv):
            continue
        unique_parts.append(p_i)

        uniq = True
        for j in range(len(unique_parts)):
            p_j = unique_parts[j]
            if polygons_equiv(p_i, p_j):
                uniq = False
                break
        if uniq:
            unique_parts.append(p_i)

    return unique_parts


def polygons_equiv(p1, p2):
    p2_first_pt = p2[0, 0]
    dist0 = cv2.pointPolygonTest(p1, (p2_first_pt[0], p2_first_pt[1]), True)
    if abs(dist0) >= 4:
        return False
    p2_last_pt = p2[-1, 0]
    dist_last = cv2.pointPolygonTest(p1, (p2_last_pt[0], p2_last_pt[1]), True)
    if abs(dist_last) >= 4:
        return False
    p2_middle_pt = p2[len(p2) // 2, 0]
    dist_middle = cv2.pointPolygonTest(p1, (p2_middle_pt[0], p2_middle_pt[1]), True)
    return abs(dist_middle) < 4


def split_2(contour):
    # сразу сравнивать с эталонным элипсом???
    pts = contour.measurements().approx_points
    angles, pts_unwrapped = geometry.compute_angles_vectorized(pts)

    # (45. < angles) & (angles <= 160.)
    # mask = angles > 160
    mask = 45 < angles
    indexes = np.where(mask)[0]
    if indexes.size == 0:
        return np.empty((0, 2)), np.empty((0, 1, 2))
    tails = pts_unwrapped[indexes]

    parts = [None] * len(indexes)

    shift = indexes[0]
    pts = np.roll(pts, -shift, axis=0)
    indexes = np.append(indexes - shift, len(pts) - 1)
    for i in range(len(indexes) - 1):
        index = indexes[i]
        next_index = indexes[i + 1]
        parts[i] = pts[index: next_index + 1]

    return tails, parts


def main():
    source = 'd:/DiskE/Computer_Vision_Task/Video_2.mp4'
    video = VideoCapture(source)
    image = video.read_at_pos(225)

    bgr = cv2.GaussianBlur(image, (3, 3), 0, dst=image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, 255, edges=gray)

    # edges = test_image()

    im = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # draw_contours([contour], im, color=(255, 0, 0), draw_measurements=False)

    corners = cv2.goodFeaturesToTrack(edges, 500, 0.01, 10)

    for x, y in corners[:, 0]:
        cv2.circle(im, (int(x), int(y)), 1, (0, 255, 0), thickness=-1)

    cv2.imshow('im', im)
    cv2.waitKey()
    # TODO: тестировать на эллипсе


main()
import cv2
import numpy as np

import geometry_utils as geometry


class MeanColorFilter:
    __dist_threshold = 35
    __squared_dist_threshold = __dist_threshold * __dist_threshold

    def __init__(self, calibrator):
        self.__calibrator = calibrator

    def filter_polygons(self, polygons, frame_image):
        # is convex poly??
        result = []
        mask = np.zeros(frame_image.shape[0:2], dtype=np.uint16)
        ref_mean_color = self.__calibrator.reference_ellipse.mean_color
        for i, poly in enumerate(polygons):
            polygon_label = i + 1
            cv2.fillConvexPoly(mask, poly.points, polygon_label)
            rr, cc = np.where(mask == polygon_label)
            polygon_mean_color = np.mean(frame_image[rr, cc], axis=0, dtype=np.float32)
            # TODO: try perform vectorized distance computation - vectorized_dist(ref_mean_color, vector_of_mean_colors)
            squared_dist = geometry.squared_color_distance(polygon_mean_color, ref_mean_color)
            if squared_dist < self.__squared_dist_threshold:
                result.append(poly)
        return result

    def filter_ellipses(self, ellipses, frame_image):
        ref_mean_color = self.__calibrator.reference_ellipse.mean_color
        ellipses_mean_colors = [ellipse_axes_mean_color(el, frame_image) for el in ellipses]
        return _items_close_to_color(ellipses, ellipses_mean_colors, ref_mean_color,
                                     self.__squared_dist_threshold, self.__dist_threshold)


def ellipse_axes_mean_color(ellipse, frame_image):
    cc, rr = ellipse.fit_ellipse.main_axes_pts
    return np.mean(frame_image[rr, cc], axis=0, dtype=np.float32)


def _items_close_to_color(items, items_colors, ref_color, squared_color_distance_threshold, color_distance_threshold):
    # TODO: м.б. будет быстрее для начала сравнивать разницы в координатах - для отсеивания явно "далеких" элементов.
    #  Так: is_too_far = abs(color_coord1 - color_coord2) > color_distance_threshold
    #  И только потом (для оставшихся) вычислять квадрат цветового расстояния
    squared_distances = geometry.squared_color_distances(ref_color, items_colors)
    indexes = np.where(squared_distances < squared_color_distance_threshold)
    return [items[i] for i in indexes[0]]

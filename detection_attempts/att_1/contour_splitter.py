import numpy as np
import geometry_utils as geometry
import utils
from detection_attempts.att_1.polygon import Polygon


class ContourSplitter:
    @staticmethod
    def __normalize(vectors):
        squared_coords = np.multiply(vectors, vectors)
        lens = np.sqrt(squared_coords[:, 0] + squared_coords[:, 1]).reshape((-1, 1))  # [1, 2, 3] => [[1],[2],[3]]
        return np.divide(vectors, lens)

    @staticmethod
    def __relative_rotation_angles(vectors):
        if len(vectors) < 2:
            return np.empty((0, 2), dtype=np.float32)
        vectors = ContourSplitter.__normalize(vectors)
        dst_vectors = vectors[1:]  # angles computation starts from second vector
        src_vectors = vectors[0:len(vectors) - 1]  # prev vectors

        # from prev to current -> src is prev, dst is current
        sin = dst_vectors[:, 1] * src_vectors[:, 0] - dst_vectors[:, 0] * src_vectors[:, 1]
        rad = np.arcsin(sin)
        return np.degrees(rad), rad

    @staticmethod
    def __is_sign_change(arg1, arg2):
        return arg1 < 0 < arg2 or arg1 > 0 > arg2

    @staticmethod
    def __points_to_vectors(points):
        if len(points.shape) == 3:
            points = points[:, 0]  # [ [[x, y]]...[[x, y]] ] -> [ [x,y]...[x,y] ]
        current_points = points[1:]
        prev_points = points[0:len(points) - 1]
        vectors = current_points - prev_points
        return vectors

    @staticmethod
    def split_by_rotation_change_points(polygon_points):
        if len(polygon_points) < 4:
            return [polygon_points]

        vectors = ContourSplitter.__points_to_vectors(polygon_points)

        angles, _ = ContourSplitter.__relative_rotation_angles(vectors)
        result_polygons = []
        current_polygon = list(polygon_points[:3])  # first 3 points

        # TODO: ignore small changes
        for i, pt in enumerate(polygon_points[3:len(polygon_points) - 1]):
            pt_index = i + 3
            angle_index = i + 2
            current_polygon.append(pt)
            angle = angles[angle_index]
            prev_angle = angles[angle_index - 1]
            if ContourSplitter.__is_sign_change(prev_angle, angle):
                result_polygons.append(np.array(current_polygon))
                current_polygon = [polygon_points[pt_index - 1], pt]  # init new part with prev point

        current_polygon.append(polygon_points[-1])
        result_polygons.append(np.array(current_polygon))

        return result_polygons

    @staticmethod
    def split_contour(contour, max_angle):
        # сразу сравнивать с эталонным элипсом???
        pts = contour.measurements().approx_points
        angles = contour.measurements().approx_points_angles

        mask = max_angle <= angles
        indexes = np.where(mask)[0]
        if indexes.size == 0:
            return [Polygon(pts)]

        parts = []

        for i in range(len(indexes) - 1):
            # todo: м.б. формировать список уникальных контуров в этом цикле??
            index = indexes[i]
            next_index = indexes[i + 1]
            if (next_index - index) < 2:  # пропускаем фрагменты из 2 и менее точек
                continue
            part_pts = pts[index: next_index + 1]
            parts.append(Polygon(part_pts))

        part_pts = np.append(pts[indexes[-1]:], pts[:indexes[0] + 1], axis=0)
        if len(part_pts) > 2:  # пропускаем фрагменты из 2 и менее точек
            parts.append(Polygon(part_pts))

        if len(parts) == 0:
            return []

        # DEBUG: uncomment
        # _parts = []
        # for p in parts:
        #     if len(p.points) < 4:
        #         _parts.append(p)
        #     else:
        #         splits = ContourSplitter.split_by_rotation_change_points(p.points)
        #         _parts.extend([Polygon(pts) for pts in splits])
        # parts = _parts

        ############################
        unique_parts = [parts[0]]
        for i in range(1, len(parts)):
            part = parts[i]
            uq_part, uq_part_index = \
                utils.first_or_default(unique_parts, lambda uniq_part: uniq_part.is_equivalent(part, 4))
            if uq_part is None:
                unique_parts.append(part)
            elif part.arc_len > uq_part.arc_len:  # compare sizes and use bigger part
                unique_parts[uq_part_index] = part

        return unique_parts


if __name__ == '__main__':
    def main():
        polygon = np.array([[5, 3], [5, 2], [6, 1], [7, 1], [8, 0]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 2
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1], [7, 1]])
        np.testing.assert_array_equal(splits[1], [[6, 1], [7, 1], [8, 0]])
        # -------------------------------
        polygon = np.array([[5, 3], [5, 2], [6, 1]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 1
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1]])
        # -------------------------------
        polygon = np.array([[5, 3], [5, 2], [6, 1], [7, 1]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 1
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1], [7, 1]])
        # ------------------------------------------
        polygon = np.array([[5, 3], [5, 2], [6, 1], [7, 1], [8, 2]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 1
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1], [7, 1], [8, 2]])
        # --------------------------------------------
        polygon = np.array([[5, 3], [5, 2], [6, 1], [7, 1], [8, 0], [7, 0]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 2
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1], [7, 1]])
        np.testing.assert_array_equal(splits[1], [[6, 1], [7, 1], [8, 0], [7, 0]])
        # ----------------------------------
        polygon = np.array([[5, 3], [5, 2], [6, 1], [7, 1], [8, 0], [9, 0]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 3
        np.testing.assert_array_equal(splits[0], [[5, 3], [5, 2], [6, 1], [7, 1]])
        np.testing.assert_array_equal(splits[1], [[6, 1], [7, 1], [8, 0]])
        np.testing.assert_array_equal(splits[2], [[7, 1], [8, 0], [9, 0]])
        # ----------------------------------
        polygon = np.array([[[5, 3]], [[5, 2]], [[6, 1]], [[7, 1]], [[8, 0]], [[9, 0]]])
        splits = ContourSplitter.split_by_rotation_change_points(polygon)
        assert len(splits) == 3
        np.testing.assert_array_equal(splits[0], [[[5, 3]], [[5, 2]], [[6, 1]], [[7, 1]]])
        np.testing.assert_array_equal(splits[1], [[[6, 1]], [[7, 1]], [[8, 0]]])
        np.testing.assert_array_equal(splits[2], [[[7, 1]], [[8, 0]], [[9, 0]]])


    main()

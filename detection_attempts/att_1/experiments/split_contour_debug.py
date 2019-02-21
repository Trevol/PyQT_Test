import cv2
import numpy as np
from detection_attempts.att_1.detect_on_video import (get_capture_and_calibration_image_video2,
                                                      get_capture_and_calibration_image_video6, Detector, Calibrator)
from cv_named_window import CvNamedWindow
from detection_attempts.att_1.experiments.detect_debug import get_detector
from detection_attempts.att_1.contour import Contour
from detection_attempts.att_1.polygon import Polygon
import geometry_utils as geometry
import utils


class DEBUG:
    @staticmethod
    def show_goodFeaturesToTrack(img, ):
        img_gft = img.copy()
        corners = cv2.goodFeaturesToTrack(img_gft, maxCorners=30, qualityLevel=0.01, minDistance=5)
        for ((x, y),) in corners.astype(np.uint32):
            img_gft[y - 1:y + 1, x - 1:x + 1] = 127
        CvNamedWindow('gft', flags=cv2.WINDOW_NORMAL).imshow(img_gft)

    @staticmethod
    def walk_points_sequence(points, is_closed=False, point_action=None):
        point_action = point_action or (lambda i, pt: print(f'{i}, {pt}'))

        max_x = points[..., 0].max() + 20
        max_y = points[..., 1].max() + 20

        img = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.polylines(img, [points], is_closed, 255)

        # show_goodFeaturesToTrack(img)

        cv2.circle(img, tuple(points[0, 0]), 3, 255, 1)
        cv2.circle(img, tuple(points[-1, 0]), 3, 127, -1)

        wnd = CvNamedWindow(flags=cv2.WINDOW_NORMAL)
        wnd.imshow(img)
        if cv2.waitKey() == 27:
            return

        for i, pt in enumerate(points):
            point_action(i, pt)
            im = img.copy()
            cv2.circle(im, tuple(pt[0]), 2, 255, 1)
            wnd.imshow(im)
            if cv2.waitKey() == 27:
                return

    @staticmethod
    def show_contour_parts(contour, parts):
        points = contour.points()
        max_x = points[..., 0].max() + 20
        max_y = points[..., 1].max() + 20
        separator = np.full((max_y, 2), 127, dtype=np.uint8)

        images = []
        for poly in parts:
            img = np.zeros((max_y, max_x), dtype=np.uint8)
            poly_points = poly.points
            if len(poly_points.shape) == 3:
                poly_points = poly_points[:, 0]
            cv2.polylines(img, [poly_points], False, 255, 1)
            for (x, y) in poly_points:
                r = 2
                img[y - r:y + r, x - r:x + r] = 200
            images.extend([img, separator])

        wnd = CvNamedWindow('parts')
        if len(images) == 0:
            return
        wnd.imshow(np.hstack(images))


def get_polygon_of_interest():
    # def __contour(img, region, len):
    #     (r1, r2), (c1, c2) = region
    #     img = img[r1:r2, c1:c2]
    #     contours = Contour.find(img)
    #     return [c for c in contours if c.len() == len][0]
    #
    # video, calibration_image = get_capture_and_calibration_image_video2()
    # frame = video.read_at_pos(624)
    #
    # # contour = __contour(frame, [(268, 389), (650, 724)], 620)
    # # contour = __contour(frame, [(268, 389), (650, 724)], 67)
    # # contour = __contour(frame, [(271, 338), (661, 722)], 369)
    # contour = __contour(frame, [(367, 438), (595, 666)], 268)
    #
    # contour_parts = ContourSplitter.split_contour(contour, 36)
    # return max(contour_parts, key=lambda p: p.len)
    pass


class ContourSplitter:
    @staticmethod
    def __normalize(vectors):
        squared_coords = np.multiply(vectors, vectors)
        lens = np.sqrt(squared_coords[:, 0] + squared_coords[:, 1]).reshape((-1, 1))  # [1, 2, 3] => [[1],[2],[3]]
        return np.divide(vectors, lens)

    @staticmethod
    def relative_rotation_angles(vectors):
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

        angles, _ = ContourSplitter.relative_rotation_angles(vectors)
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

    #############################################
    @staticmethod
    def array_roll_forward(a):
        # np.roll(a, 1, axis=0)
        last = a[-1:]
        except_last = a[:-1]
        return np.concatenate((last, except_last))

    @staticmethod
    def array_roll_backward(a):
        # np.roll(a, -1, axis=0)
        except_first = a[1:]
        first = a[:1]
        return np.concatenate((except_first, first))

    @staticmethod
    def get_contour_unit_vectors(contour_pts):
        if len(contour_pts.shape) == 3:
            contour_pts = contour_pts[:, 0]  # [ [[x, y]]...[[x, y]] ] -> [ [x,y]...[x,y] ]
        next_pts = ContourSplitter.array_roll_backward(contour_pts)  # np.roll(contour_pts, -1, axis=0)
        prev_pts = ContourSplitter.array_roll_forward(contour_pts)  # np.roll(contour_pts, 1, axis=0)

        input_vectors = ContourSplitter.__normalize(contour_pts - prev_pts)
        output_vectors = ContourSplitter.__normalize(next_pts - contour_pts)
        return input_vectors, output_vectors

    @staticmethod
    def get_contour_angles(contour_pts):
        inp_vecs, out_vecs = ContourSplitter.get_contour_unit_vectors(contour_pts)
        inp_vecs_x = inp_vecs[:, 0]
        inp_vecs_y = inp_vecs[:, 1]
        out_vecs_x = out_vecs[:, 0]
        out_vecs_y = out_vecs[:, 1]

        # vec_angle_cos = scalar_product of unit vectors
        vec_angle_cos = inp_vecs_x * out_vecs_x + inp_vecs_y * out_vecs_y
        vec_angle_rads = np.arccos(vec_angle_cos)

        # signed sinus value of rotation angle
        rotation_sins = out_vecs_y * inp_vecs_x - out_vecs_x * inp_vecs_y

        return vec_angle_rads, rotation_sins

    @staticmethod
    def contour_split_indexes(vec_angle_rads, rotation_sins, max_angle_in_rads, tolerance):
        prev_rotation_sins = ContourSplitter.array_roll_forward(rotation_sins)
        assert len(vec_angle_rads) == len(rotation_sins) == len(prev_rotation_sins)

        # where change in sign (rotation direction)?
        # raise NotImplementedError('where change in sign (rotation direction)?')

        print(rotation_sins)
        print(prev_rotation_sins)

        indexes = []
        for i, (angle, rot_sin, prev_rot_sin) in enumerate(zip(vec_angle_rads, rotation_sins, prev_rotation_sins)):
            if angle > max_angle_in_rads or (rot_sin < 0 < prev_rot_sin) or (rot_sin > 0 > prev_rot_sin):
                indexes.append(i)
        return indexes

    @staticmethod
    def split_contour_by_indexes(contour_pts, indexes):
        if len(indexes) == 0:
            return [contour_pts]
        parts = []
        for i, index in enumerate(indexes[1:]):
            prev_index = indexes[i]  # i is prev index because we start from second item
            parts.append(contour_pts[prev_index:index + 1])
        # from index to last and from first to index_0
        parts.append(np.concatenate((contour_pts[indexes[-1]:], contour_pts[:indexes[0] + 1])))
        return parts

    @staticmethod
    def split_contour_new(contour_pts, max_angle_in_rads, tolerance):
        # max angle
        # change in direction
        # ignore small changes

        # compute unit-form vectors
        # compute scalar product
        # compute signed rotation angle (scalar product gives angle)

        cls = ContourSplitter
        vec_angle_rads, rotation_sins = cls.get_contour_angles(contour_pts)
        split_indexes = cls.contour_split_indexes(vec_angle_rads, rotation_sins, max_angle_in_rads, tolerance)
        parts = cls.split_contour_by_indexes(contour_pts, split_indexes)
        return parts


def main():
    contour = np.array(
        [[[56, 0]], [[54, 6]], [[46, 17]], [[32, 30]], [[29, 30]], [[18, 37]], [[0, 39]], [[18, 37]], [[30, 30]],
         [[33, 30]], [[46, 18]], [[54, 7]]])
    max_angle_in_rads = np.radians(33.258029614706885)

    parts = ContourSplitter.split_contour_new(contour, max_angle_in_rads, 10)
    DEBUG.show_contour_parts(Contour(contour), [Polygon(p) for p in parts])
    cv2.waitKey()

    # c = Contour(contour)
    # pp = ContourSplitter.split_contour(c, 53.26)
    # DEBUG.show_contour_parts(c, pp)
    # cv2.waitKey()

def main():
    contour = np.array(
        [[[56, 0]], [[54, 6]], [[46, 17]], [[32, 30]], [[29, 30]], [[18, 37]], [[0, 39]], [[18, 37]], [[30, 30]],
         [[33, 30]], [[46, 18]], [[54, 7]]])

    vec_angle_rads, rot_sins = ContourSplitter.get_contour_angles(contour)
    prev_rot_sins = ContourSplitter.array_roll_forward(rot_sins)
    vec_angle_degrees = np.degrees(vec_angle_rads)

    def print_point_info(i, pt):
        vect_angle_info = f'vector_angle_rads/deg:{vec_angle_rads[i]}/{vec_angle_degrees[i]}'
        print(f'{i}: {pt}.', vect_angle_info, rot_sins[i], prev_rot_sins[i])

    DEBUG.walk_points_sequence(contour, True, print_point_info)



if __name__ == '__main__':
    main()

# def main_DEBUG():
#     contour = np.array(
#         [[[56, 0]], [[54, 6]], [[46, 17]], [[32, 30]], [[29, 30]], [[18, 37]], [[0, 39]], [[18, 37]], [[30, 30]],
#          [[33, 30]], [[46, 18]], [[54, 7]]])
#     max_angle_in_rads = np.radians(33.258029614706885)
#
#     vector_angle_radians, rotation_direction, (input_vectors, output_vectors) = \
#         ContourSplitter.split_contour_new(contour, max_angle_in_rads)
#     vector_angle_degrees = np.degrees(vector_angle_radians)
#
#     def print_point_info(i, pt):
#         vect_angle_info = f'vector_angle_cos/rads/deg:{vector_angle_radians[i]}/{vector_angle_degrees[i]}'
#         print(f'{i}: {pt}.', f'{input_vectors[i]}->{output_vectors[i]}', vect_angle_info, rotation_direction[i])
#
#     # DEBUG.walk_points_sequence(contour, True, print_point_info)
#
#     c = Contour(contour)
#     pp = ContourSplitter.split_contour(c, 33.26)
#     DEBUG.show_contour_parts(c, pp)
#     cv2.waitKey()

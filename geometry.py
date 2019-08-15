import numpy as np
import math
import cv2


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def points2line(pt1, pt2):
    line = np.empty((0, 2), int)
    coordinate = np.reshape(np.asarray(pt1), (1, 2))
    line = np.append(line, np.array(coordinate), axis=0)
    coordinate = np.reshape(np.asarray(pt2), (1, 2))
    line = np.append(line, np.array(coordinate), axis=0)
    return line


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.asarray([x, y]).astype(int)


def verticalLine(line_pt1, line_pt2, pt3, half_length):
    direction_vector = np.asarray([line_pt2[0] - line_pt1[0], line_pt2[1] - line_pt1[1]])
    magnitude = math.hypot(direction_vector[0], direction_vector[1])
    norm_direction_vector = direction_vector / magnitude
    rotated_norm_direction_vector = np.asarray([-norm_direction_vector[1], norm_direction_vector[0]])
    pt4 = pt3 + half_length * rotated_norm_direction_vector
    pt5 = pt3 - half_length * rotated_norm_direction_vector
    return np.asarray([pt4, pt5]).astype(int)


def scaled_line(pt1, pt2, scale):
    pt = [pt1[0] + scale * (pt2[0] - pt1[0]), pt1[1] + scale * (pt2[1] - pt1[1])]
    return np.asarray([pt[0], pt[1]]).astype(int)


def bisector(line1, line2, half_length=20, cross_point=np.nan):
    theta_u = math.atan2(line1[1][1] - line1[0][1], line1[1][0] - line1[0][0])
    theta_v = math.atan2(line2[1][1] - line2[0][1], line2[1][0] - line2[0][0])
    middle_theta = (theta_u + theta_v) / 2
    bisector_vector_diretion = np.asarray([math.cos(middle_theta), math.sin(middle_theta)])
    if np.isnan(cross_point.any()):
        cross_point = line_intersection(line1, line2)
    bisector_vector = np.asarray([cross_point - half_length * bisector_vector_diretion,
                                  cross_point + half_length * bisector_vector_diretion]).astype(int)
    # Need to be rotated 90 degrees for true output
    bisector_vector = verticalLine(bisector_vector[0, :], bisector_vector[1, :], cross_point, half_length)
    return bisector_vector


def in_area(im, points_polygon, test_points):
    src = np.zeros(im.shape[0:2], dtype=np.uint8)
    # Create a sequence of points to make a contour
    # Draw it in src
    cv2.polylines(src, [points_polygon], True, (255), thickness=1, lineType=cv2.LINE_AA)
    # Get the contours
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the distances to the contour
    result = np.empty((test_points.shape[0], 1))
    for test_point_index in range(test_points.shape[0]):
        result[test_point_index] = cv2.pointPolygonTest(contours[0], (
        test_points[test_point_index, 0], test_points[test_point_index, 1]), False)
    return result

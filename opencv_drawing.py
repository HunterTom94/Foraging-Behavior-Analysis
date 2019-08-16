import numpy as np
import cv2

def LinkPoints(im, pt1, pt2, BGR=(255, 200, 0)):
    points_np = np.empty((0, 2), int)
    points_np = np.append(points_np, np.reshape(pt1, (1, 2)), axis=0)
    points_np = np.append(points_np, np.reshape(pt2, (1, 2)), axis=0)
    cv2.polylines(im, [points_np], False, BGR, thickness=1, lineType=cv2.LINE_AA)


def drawPolyline(im, points, isClosed=False, BGR=(255, 200, 0)):
    cv2.polylines(im, [points], isClosed, BGR, thickness=1, lineType=cv2.LINE_AA)


def drawSegmentPoints(im, segment_points):
    for point_index in range(segment_points.shape[0]):
        cv2.circle(im, (segment_points[point_index, 0], segment_points[point_index, 1]), 2, (0, 255, 0), 1,
                   cv2.LINE_AA)
    return im

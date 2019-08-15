import numpy as np
import cv2
import pandas as pd
from scipy import optimize
import os
import itertools
import math
from geometry import angle_between
from opencv_drawing import LinkPoints
from matplotlib import cm

scale = 15


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def leastsq_circle(x, y):
    def calc_R(x, y, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c, x, y):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(x, y, *c)
        return Ri - Ri.mean()

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R, residu


def save_trajectory(data_name, sheet):
    data = pd.read_excel(data_name, sheet_name=sheet,
                         header=None,
                         index_col=False,
                         keep_default_na=True
                         )

    for fly_ind in range(data.shape[1] // 3):
        data.iloc[3:, fly_ind * 3:(fly_ind + 1) * 3].dropna().to_pickle(
            'trajectories\\{}_{}.pkl'.format(sheet, fly_ind))


# save_trajectory('data.xlsx','60hr')

def circle_fit_selection(trajectory):
    scale = 15
    trajectory.iloc[:, 0:2] = trajectory.iloc[:, 0:2].astype(float).round(0).astype(int)
    coord_x = np.asarray(trajectory.iloc[:, 0] - trajectory.iloc[:, 0].min()).astype(int) + 1
    coord_y = np.asarray(trajectory.iloc[:, 1] - trajectory.iloc[:, 1].min()).astype(int) + 1
    pts = np.stack((coord_x, coord_y), axis=1)
    big_pts = pts * scale

    pad = np.zeros((np.max(coord_y + 1), np.max(coord_x + 1)), dtype=np.uint8)
    # pad[coord_y,coord_x] = 1

    pad = cv2.resize(pad, (pad.shape[0] * scale, pad.shape[1] * scale))
    cv2.polylines(pad, [big_pts], color=(255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(pad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_pts = contours[0].reshape((contours[0].shape[0], contours[0].shape[2]))
    contour_pad = np.zeros(pad.shape, dtype=np.uint8)
    # cv2.drawContours(contour_pad, contours, 0, (255), 3)
    cv2.polylines(contour_pad, [contour_pts], color=(255), isClosed=True, thickness=1, lineType=cv2.LINE_AA)

    # exit()

    x = contour_pts[:, 1]
    y = contour_pts[:, 0]

    yc, xc, R, residu = leastsq_circle(x, y)
    area = cv2.contourArea(contours[0])
    if area > 580000 and area < 620000 and np.sqrt(residu) < 300:
        return True
    else:
        return False


def line_generator(trajectory):
    trajectory.iloc[:, 0:2] = trajectory.iloc[:, 0:2].astype(float).round(0).astype(int)
    diff_time = np.diff(trajectory.iloc[:, 2])
    coord_x = np.asarray(trajectory.iloc[:, 0] +31).astype(int)
    coord_y = np.asarray(trajectory.iloc[:, 1] +31).astype(int)
    pts = np.stack((coord_x, coord_y), axis=1)
    big_pts = pts * scale

    lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'angle', 'length', 'speed', 'time', 'start_time'])
    for ind, (pt1, pt2) in enumerate(pairwise(big_pts)):
        dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        if ind == 0:
            angle = np.nan
        elif dist == 0:
            angle = np.nan
        else:
            last_line = ind - 1
            while lines.loc[last_line, 'length'] == 0:
                last_line -= 1
                if last_line < 0:
                    angle = np.nan
                    break

            angle = angle_between(pt2 - pt1, lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1'])
        lines = lines.append(
            {'line_ID': ind, 'point1': pt1, 'point2': pt2, 'angle': angle, 'length': dist,
             'speed': dist / diff_time[ind], 'time': diff_time[ind], 'start_time': trajectory.iloc[ind, 2]},
            ignore_index=True)
        return lines


def draw_trajectory(lines):
    pad = np.zeros((72*scale, 72*scale), dtype=np.uint8)
    color_ls = []
    for index in range(256):
        color_ls.append(cm.jet(index)[:3])

    for index, row in lines.iterrows():
        LinkPoints(pad, row['point1'],row['point2'], RGB=color_ls)


    # cv2.imshow('test',pad)
    # cv2.waitKey(-1)


# directory = os.fsencode('trajectories')
#
# for file in os.listdir(directory):
#      filename = os.fsdecode(file)
#      if filename.endswith(".pkl"):
#          sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
#          if circle_fit_selection(sample):
#              sample.to_pickle('selected_trajectories\\{}'.format(filename))

sample = pd.read_pickle('selected_trajectories\\12hr_2.pkl')
draw_trajectory(line_generator(sample))

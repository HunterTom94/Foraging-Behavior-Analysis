import numpy as np
import cv2
import pandas as pd
from scipy import optimize, stats
import os
import itertools
import math
from geometry import angle_between
from opencv_drawing import LinkPoints
from matplotlib import cm
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

scale = 10
edge_size = 2.5
line_template = pd.DataFrame(
    columns=['line_ID', 'point1', 'point2', 'angle', 'length', 'speed', 'distance', 'time', 'start_time',
             'on_edge', 'trajectory_name'])


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

def circle_fit(pts):
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    cv2.polylines(pad, [pts], color=(255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(pad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_pts = contours[0].reshape((contours[0].shape[0], contours[0].shape[2]))
    # contour_pad = np.zeros(pad.shape, dtype=np.uint8)
    # cv2.drawContours(contour_pad, contours, 0, (255), 3)
    # cv2.polylines(contour_pad, [contour_pts], color=(255), isClosed=True, thickness=1, lineType=cv2.LINE_AA)

    x = contour_pts[:, 1]
    y = contour_pts[:, 0]

    yc, xc, R, residu = leastsq_circle(x, y)
    area = cv2.contourArea(contours[0])
    return yc, xc, R, residu, area


def circle_fit_selection(trajectory):
    pts = trajectory2pts(trajectory)
    yc, xc, R, residu, area = circle_fit(pts)
    if area > 580000 and area < 620000 and np.sqrt(residu) < 300:
        return True
    else:
        return False


def trajectory2pts(trajectory):
    trajectory.iloc[:, 0:2] = trajectory.iloc[:, 0:2].astype(float).round(0).astype(int)

    coord_x = np.asarray(trajectory.iloc[:, 0] + 35).astype(int)
    coord_y = np.asarray(trajectory.iloc[:, 1] + 35).astype(int)
    pts = np.stack((coord_x, coord_y), axis=1)
    big_pts = pts * scale
    return big_pts


def line_generator(trajectory, pts, trajectory_name):
    diff_time = np.diff(trajectory.iloc[:, 2])

    yc, xc, R, _, _ = circle_fit(pts)

    lines = line_template
    for ind, (pt1, pt2) in enumerate(pairwise(pts)):
        length = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        if ind == 0:
            angle = np.nan
        elif length == 0:
            angle = np.nan
        else:
            last_line = ind - 1
            while lines.loc[last_line, 'length'] == 0:
                last_line -= 1
                if last_line < 0:
                    angle = np.nan
                    break
                angle = angle_between(pt2 - pt1, lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1'])

        dist = math.hypot(pt1[0] - yc, pt1[1] - xc)
        if dist >= (R - edge_size * scale):
            if math.hypot(pt2[0] - yc, pt2[1] - xc) >= (R - edge_size * scale):
                on_edge = 1
            else:
                on_edge = 0
        else:
            on_edge = 0
        # if dist>(R-edge_size*scale) and dist < (R+edge_size*scale):
        #     if math.hypot(pt2[0] - yc, pt2[1] - xc)>(R-edge_size*scale) and math.hypot(pt2[0] - yc, pt2[1] - xc) < (R+edge_size*scale):
        #         on_edge = 1
        #     else:
        #         on_edge = 0
        # else:
        #     on_edge = 0
        lines = lines.append(
            {'line_ID': ind, 'point1': pt1, 'point2': pt2, 'angle': angle, 'length': length,
             'speed': length / diff_time[ind], 'distance': dist, 'time': diff_time[ind],
             'start_time': trajectory.iloc[ind, 2], 'on_edge': on_edge, 'trajectory_name': trajectory_name},
            ignore_index=True)
    return lines


def draw_trajectory(lines):
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
    color_ls = []
    for index in range(256):
        color_ls.append(cm.jet(index)[:3])
    color_ls = np.flip(np.asarray(color_ls), axis=1) * 255

    # 'angle', 'length', 'speed', 'distance', 'time', 'start_time'
    data_to_plot = lines['angle']
    inds = np.digitize(data_to_plot.values / (data_to_plot.max() / 255), np.arange(256)) - 1

    for index, (_, row) in enumerate(lines.iterrows()):
        LinkPoints(pad, row['point1'], row['point2'], BGR=tuple(color_ls[inds[index]].tolist()))

    pad_copy = pad.copy()
    frame_index = 0

    while True:
        pad = pad_copy

        cv2.circle(pad,(lines.loc[frame_index,'point1'][1],lines.loc[frame_index,'point1'][0]), 3, (255,255,255),-1,cv2.LINE_AA)
        cv2.imshow('test', pad)
        input = cv2.waitKeyEx(-1)
        if input == 2424832: # Left Arrow Key
            frame_index -= 1
            if frame_index < 0:
                frame_index = 0
        elif input == 2555904: # Right Arrow Key
            frame_index += 1
            if frame_index > lines.shape[0]:
                frame_index = lines.shape[0]
        print(frame_index)


sample = pd.read_pickle('selected_trajectories\\60hr_7.pkl')
pts = trajectory2pts(sample)
lines = line_generator(sample, pts, '60hr_7')
lines = lines[pd.notnull(lines['angle'])]
lines = lines[pd.notnull(lines['distance'])]
lines = lines[lines['on_edge'] == 0]
lines = lines.reset_index(drop=True)
draw_trajectory(lines)
exit()
pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
cv2.polylines(pad, [pts], color=(255, 255, 255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)
yc, xc, R, residu, area = circle_fit(pts)
cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
           int(3), color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
           np.round((R - edge_size * scale), 0).astype(int), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
           np.round(R, 0).astype(int), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
# cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
#            np.round((R + edge_size * scale), 0).astype(int), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow('test1', pad)
cv2.waitKey(-1)

exit()

# directory = os.fsencode('trajectories')
#
# for file in os.listdir(directory):
#      filename = os.fsdecode(file)
#      if filename.endswith(".pkl"):
#          sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
#          if circle_fit_selection(sample):
#              sample.to_pickle('selected_trajectories\\{}'.format(filename))


# lines = pd.read_pickle('60hr_lines.pkl')
sample = pd.read_pickle('selected_trajectories\\60hr_7.pkl')
pts = trajectory2pts(sample)
lines = line_generator(sample, pts, '60hr_7')
lines = lines[pd.notnull(lines['angle'])]
lines = lines[pd.notnull(lines['distance'])]
lines = lines[lines['on_edge'] == 0]

angle = lines['angle'].values
distance = lines['distance'].values

sns.scatterplot(angle, distance)
# sns.set_style('darkgrid')
# sns.distplot(data)
# sns.distplot(data, fit=stats.laplace, kde=False)
plt.show()
exit()

start_time = time()
condition = '60hr'
count = 0
total_lines = line_template

directory = os.fsencode('selected_trajectories')

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".pkl") and filename.startswith(condition):
        print(count)
        count += 1
        sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
        total_lines = total_lines.append(line_generator(sample, trajectory2pts(sample), filename.split('.')[0]),
                                         ignore_index=True)
        print(int(time() - start_time))

total_lines.to_pickle('{}_lines.pkl'.format(condition))

import numpy as np
import cv2
import pandas as pd
from scipy import optimize, stats
import os
import itertools
import math
from geometry import signed_angle_between, angle_between, in_area
from opencv_drawing import LinkPoints, drawPolyline
from matplotlib import cm
from time import time
from data_management import condition_filter
from joblib import Parallel, delayed
import multiprocessing
from numbers import Number
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

scale = 10
edge_size = 4
line_template = pd.DataFrame(
    columns=['line_ID', 'point1', 'point2', 'angle', 'length', 'speed', 'distance', 'time', 'start_time',
             'on_edge', 'condition', 'trajectory_index'])
gaus_x = np.linspace(-100, 100, 20000)
thres_ind = np.argwhere(gaussian(gaus_x, 0, 0.5) > 0.0001)[0]
assert thres_ind > 0
gaus_thres_sig0p5 = gaus_x[thres_ind]


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def leastsq_circle(x, y):

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


def circle_fit(pts, recur_flag=0):
    def far_from_center(pt, center, threshold):
        return math.hypot(pt[0] - center[0], pt[1] - center[1]) > threshold
    def find_circle_enclose(x_estimate, y_estimate, contour_pts, delete_width):
        def row_isin_array(row, array):
            return bool(np.apply_along_axis(lambda i: np.array_equal(i, row), 1, array).sum())
        def in_circle(center_offset, R):
            def points_in_circle_np(radius, x0=0, y0=0, ):
                x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
                y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
                x, y = np.where((np.hypot((x_ - x0)[:, np.newaxis], y_ - y0) <= radius))  # alternative implementation
                for x, y in zip(x_[x], y_[y]):
                    yield x, y
            center = center_offset
            circle_pts = np.asarray(list(points_in_circle_np(R, center[0], center[1])))
            return np.all(in_area(fit_pad.copy(), circle_pts, contour_pts_nocenter).astype(int) == 1).astype(int)


        contour_pts_nocenter = contour_pts[
            (contour_pts[:, 0] < (x_estimate - delete_width* scale) ) | (contour_pts[:, 0] > (x_estimate + delete_width* scale) )]
        fit_pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
        fit_pad_copy = fit_pad.copy()

        D = pdist(contour_pts_nocenter)
        D = squareform(D)
        N, [I_row, I_col] = nanmax(D), unravel_index(argmax(D), D.shape)
        R = int(N/2)+1
        if R < 295:
            R = 295

        away_from_center = 10
        while R < 35 * scale:
            # print(R)
            last_position = np.array([])
            center_offset = 0
            while center_offset < away_from_center:
                # print(center_offset)
                x = np.arange(-center_offset, center_offset+1)
                y = np.arange(-center_offset, center_offset + 1)
                X, Y = np.meshgrid(x, y)
                positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
                if center_offset:
                    updated_positions = positions[~np.apply_along_axis(row_isin_array, 1, positions, last_position), :]
                else:
                    updated_positions = positions

                last_position = positions
                updated_positions = updated_positions[np.argsort(np.abs(updated_positions).sum(axis=1)), :]
                updated_positions = (np.array([x_estimate, y_estimate]) + updated_positions *scale).astype(int)
                boundary_filter_minus = updated_positions - np.array([R,R])
                updated_positions = updated_positions[~np.any(boundary_filter_minus <= 1, axis=1), :]
                boundary_filter_plus = updated_positions + np.array([R, R])
                updated_positions = updated_positions[~np.any(boundary_filter_plus >= 699, axis=1), :]


                # For Demonstration
                # for row in range(updated_positions.shape[0]):
                #     fit_pad = fit_pad_copy.copy()
                #     cv2.polylines(fit_pad, [contour_pts_nocenter], color=(255), isClosed=True, thickness=1,
                #                   lineType=cv2.LINE_AA)
                #
                #     cv2.circle(fit_pad, (updated_positions[row,0], updated_positions[row,1]), int(R), (255), 1, cv2.LINE_AA)
                #     cv2.imshow('test', fit_pad)
                #     cv2.waitKey(-1)
                #     exit()


                if updated_positions.size != 0:
                    in_circle_result = np.apply_along_axis(in_circle, 1, updated_positions, R)
                    if np.any(in_circle_result == 1):
                        xc = updated_positions[np.nonzero(in_circle_result)[0], 0]
                        yc = updated_positions[np.nonzero(in_circle_result)[0], 1]
                        return xc[0], yc[0], R
                fit_pad = fit_pad_copy.copy()
                center_offset += 1
            R += 1

    delete_width = 10
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    cv2.polylines(pad, [pts], color=(255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(pad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour_pts_left = contour_pts[contour_pts[:, 0] < 25*scale]
    # contour_pts_right = contour_pts[contour_pts[:, 0] > 45 * scale]

    if recur_flag == 1:
        return contours

    area = cv2.contourArea(contours[0])
    contour_pts = contours[0].reshape((contours[0].shape[0], contours[0].shape[2]))
    x = contour_pts[:, 1]
    y = contour_pts[:, 0]

    y_estimate = np.mean([x.min(), x.max()])
    x_estimate = np.mean([y.min(), y.max()])

    meanR = calc_R(y, x, x_estimate, y_estimate).mean()
    outlier_index = np.apply_along_axis(far_from_center, 1, pts, [x_estimate, y_estimate], meanR + 30)
    has_outlier = bool(outlier_index.astype(int).sum())
    if has_outlier:
        # print('outlier')
        pts = pts[~outlier_index, :]
        contours = circle_fit(pts, recur_flag=1)

        area = cv2.contourArea(contours[0])
        contour_pts = contours[0].reshape((contours[0].shape[0], contours[0].shape[2]))
        x = contour_pts[:, 1]
        y = contour_pts[:, 0]

        y_estimate = np.mean([x.min(), x.max()])
        x_estimate = np.mean([y.min(), y.max()])

        # meanR = calc_R(y, x, x_estimate, y_estimate).mean()

        # outlier_index = np.apply_along_axis(far_from_center, 1, pts, [x_estimate, y_estimate], meanR + 30)
        # has_outlier = bool(outlier_index.astype(int).sum())



    contour_pad = np.zeros(pad.shape, dtype=np.uint8)
    contour_pad = cv2.cvtColor(contour_pad, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(contour_pad, contours, 0, (255), 3)
    cv2.polylines(contour_pad, [pts], color=(255, 255, 255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)
    # cv2.polylines(contour_pad, [contour_pts_left], color=(255, 255, 255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)
    # cv2.polylines(contour_pad, [contour_pts_right], color=(255, 255, 255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)

    yc, xc, R, residu = leastsq_circle(x, y)
    if not (area > 260000 and area < 280000 and np.sqrt(residu) < 200):
        (xc, yc, R) = find_circle_enclose(x_estimate, y_estimate, pts, delete_width)

    # cv2.circle(contour_pad, (int(xc), int(yc)), int(R), (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.imshow('contour_fit', contour_pad)
    # cv2.waitKey(-1)
    # exit()
    return yc, xc, R


def circle_fit_selection(trajectory):
    pts = trajectory2pts(trajectory)
    circle_fit(pts)
    exit()
    # yc, xc, R, residu, area = circle_fit(pts)
    # if area > 260000 and area < 280000 and np.sqrt(residu) < 200:
    #     return True
    # else:
    #     return False


def trajectory2pts(trajectory):
    trajectory.iloc[:, 0:2] = trajectory.iloc[:, 0:2].astype(float).round(0).astype(int)

    coord_x = np.asarray(trajectory.iloc[:, 0] + 35)
    coord_y = np.asarray(trajectory.iloc[:, 1] + 35)
    pts = np.stack((coord_x, coord_y), axis=1)
    big_pts = pts * scale
    return big_pts.astype(int)


def line_generator(trajectory, pts, trajectory_name):
    def gauss_conc(distance,sigma):
        return np.round(gaussian(distance / R * -gaus_thres_sig0p5, 0, sigma),5)

    print(trajectory_name.split('_')[0])
    print(int(trajectory_name.split('_')[1]))
    # print('start')
    diff_time = np.diff(trajectory.iloc[:, 2])

    try:
        yc, xc, R = circle_fit(pts)
    except:
        print('Skipped {}_{}'.format(trajectory_name.split('_')[0], trajectory_name.split('_')[1]))
        return

    lines = line_template
    skip_angle = 0
    for ind, (pt1, pt2) in enumerate(pairwise(pts)):
        # Calculate Length Travelled
        length = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        # Calculate turn angle
        if ind == 0:
            angle = np.nan
            bearing = np.nan
            turn_rate = np.nan
            turn_speed = np.nan
        elif length == 0:
            angle = np.nan
            bearing = np.nan
            turn_rate = np.nan
            turn_speed = np.nan
        else:
            last_line = ind - 1
            while lines.loc[last_line, 'length'] == 0:
                last_line -= 1
                if last_line < 0:
                    angle = np.nan
                    bearing = np.nan
                    turn_rate = np.nan
                    turn_speed = np.nan
                    skip_angle = 1
                    break
            if not skip_angle:
                # angle = signed_angle_between(pt2 - pt1, lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1'])
                # Negative Sign on y coordinate to be in Accordance with draw trajectory (x-y axis swap)
                angle = signed_angle_between((lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1']) * np.array([1,-1]), (pt2 - pt1) * np.array([1,-1]))
                # Calculate Bearing
                # Negative Sign on y coordinate to be in Accordance with draw trajectory (x-y axis swap)
                bearing = signed_angle_between((lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1']) * np.array([1, -1]),
                                               (np.array([xc, yc]) - pt1) * np.array([1, -1]))
                turn_rate = angle / lines.loc[last_line, 'length']
                turn_speed = angle / lines.loc[last_line, 'time']
        # Calculate Distance
        dist = math.hypot(pt1[0] - xc, pt1[1] - yc)
        # Calculate on Edge
        if dist >= (R - edge_size * scale):
            if math.hypot(pt2[0] - xc, pt2[1] - yc) >= (R - edge_size * scale):
                on_edge = 1
            else:
                on_edge = 0
        else:
            on_edge = 0

        # Calculating concentration
        conc_s0p5 = gauss_conc(dist, 0.5)[0]
        conc_s1 = gauss_conc(dist, 1)[0]
        conc_s2 = gauss_conc(dist, 2)[0]
        conc_s3 = gauss_conc(dist, 3)[0]

        lines = lines.append(
            {'line_ID': ind, 'point1': pt1, 'point2': pt2, 'angle': np.round(angle, 1),
             'turn_rate': np.round(turn_rate, 3),
             'turn_speed': np.round(turn_speed, 1), 'length': np.round(length, 1),
             'speed': np.round(length / diff_time[ind], 1), 'distance': dist,
             'time': diff_time[ind], 'start_time': trajectory.iloc[ind, 2], 'on_edge': on_edge,
             'bearing': np.round(bearing, 1), 'conc_s0p5': conc_s0p5, 'conc_s1': conc_s1, 'conc_s2': conc_s2,
             'conc_s3': conc_s3},
            ignore_index=True)
    lines['condition'] = trajectory_name.split('_')[0]
    lines['trajectory_index'] = int(trajectory_name.split('_')[1])
    lines['x_center'] = xc
    lines['y_center'] = yc
    lines['radius'] = R
    lines['delta_distance'] = np.round(np.insert(np.diff(lines['distance'].values), 0, np.nan), 1)
    lines['d_dist_rate'] = np.insert(np.round((np.diff(lines['distance'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s0p5'] = np.insert(np.round((np.diff(lines['conc_s0p5'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s1'] = np.insert(np.round((np.diff(lines['conc_s1'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s2'] = np.insert(np.round((np.diff(lines['conc_s2'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s3'] = np.insert(np.round((np.diff(lines['conc_s3'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    # print(trajectory_name.split('_')[0])
    # print(int(trajectory_name.split('_')[1]))
    # print('finish')
    return lines


def draw_trajectory(lines):
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
    color_ls = []
    for index in range(256):
        color_ls.append(cm.jet(index)[:3])
    color_ls = np.flip(np.asarray(color_ls), axis=1) * 255

    # 'angle', 'length', 'speed', 'distance', 'time', 'start_time'

    data_to_plot = 'speed'
    inds = np.digitize(lines[data_to_plot].values / (lines[data_to_plot].max() / 255), np.arange(256)) - 1

    for index, (_, row) in enumerate(lines.iterrows()):
        LinkPoints(pad, row['point1'], row['point2'], BGR=tuple(color_ls[inds[index]].tolist()))

    overlay = pad.copy()

    cv2.circle(overlay, (int(pad.shape[0]/2), int(pad.shape[1]/2)), 45, (255, 255, 255),
               -1, cv2.LINE_AA)

    alpha = 0.6

    pad = cv2.addWeighted(overlay,alpha,pad,1-alpha,0)

    pad_copy = pad.copy()
    frame_index = 0
    cv2.namedWindow('draw_pad')

    while True:
        pad = pad_copy.copy()
        cv2.polylines(pad, [np.array([[int(pad.shape[0]/2), int(pad.shape[1]/2)], [lines.loc[frame_index, 'point1'][0], lines.loc[frame_index, 'point1'][1]]])],False,(255, 255, 255),thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(pad, (lines.loc[frame_index, 'point1'][0], lines.loc[frame_index, 'point1'][1]), 3, (255, 255, 255),
                   -1, cv2.LINE_AA)
        cv2.putText(pad, 'Deviation: {}'.format(np.round(lines.loc[frame_index, 'bearing']), 1), (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        # cv2.putText(pad, 'TimeStamp: {}'.format(np.round(lines.loc[frame_index, 'start_time']), 1), (400, 680),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.putText(pad, 'TimeStamp: {}'.format(lines.loc[frame_index, 'start_time']), (400, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.putText(pad, 'Angle: {}'.format(np.round(lines.loc[frame_index, 'angle'], 1)),
                    (10, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.putText(pad, '{}: {}'.format(data_to_plot, np.round(lines.loc[frame_index, data_to_plot], 1)),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.imshow('draw_pad', pad)
        input = cv2.waitKeyEx(-1)

        if input == 2424832:  # Left Arrow Key
            frame_index -= 1
            if frame_index < 0:
                frame_index = 0
        elif input == 2555904:  # Right Arrow Key
            frame_index += 1
            if frame_index >= lines.shape[0]:
                frame_index = lines.shape[0] - 1
        elif input == 27:  # Esc Key
            cv2.destroyAllWindows()
            break

def draw_single_trajectory(lines):
    text_pad = np.zeros((70 * scale, 40 * scale), dtype=np.uint8)
    text_pad = cv2.cvtColor(text_pad, cv2.COLOR_GRAY2BGR)
    empty_text = text_pad.copy()
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)

    # Grid
    # cv2.polylines(pad, [np.array([[int(0), int(pad.shape[0]/2)],[int(pad.shape[1]), int(pad.shape[0]/2)]])],
    #                             False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    # cv2.polylines(pad, [np.array([[int(pad.shape[1] / 2), int(0)], [int(pad.shape[1] / 2), int(pad.shape[0])]])],
    #               False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


    empty_pad = pad.copy()
    color_ls = []
    for index in range(256):
        color_ls.append(cm.jet(index)[:3])
    color_ls = np.flip(np.asarray(color_ls), axis=1) * 255

    # 'angle', 'length', 'speed', 'distance', 'time', 'start_time'

    data_to_plot = 'speed'

    frame_index = 0
    trajectory_index = -1

    cv2.namedWindow('draw_pad')
    cv2.moveWindow("draw_pad", 320, 200)
    cv2.namedWindow("text_board")
    cv2.moveWindow("text_board", 1020, 200)
    curr_trajectory_index = 0
    while True:
        print(curr_trajectory_index)
        # curr_trajectory_index = lines.loc[frame_index, 'trajectory_index']
        if curr_trajectory_index != trajectory_index:
            pad = empty_pad.copy()
            trajectory_index = curr_trajectory_index
            single_trajectory = lines[lines['trajectory_index'] == trajectory_index]
            single_trajectory = single_trajectory.reset_index(drop=True)
            inds = np.digitize(single_trajectory[data_to_plot].values / (single_trajectory[data_to_plot].max() / 255), np.arange(256)) - 1
            for index, (_, row) in enumerate(single_trajectory.iterrows()):
                LinkPoints(pad, row['point1'], row['point2'], BGR=tuple(color_ls[inds[index]].tolist()))

            overlay = pad.copy()

            cv2.circle(overlay, (int(single_trajectory.iloc[0].loc['x_center']), int(single_trajectory.iloc[0].loc['y_center'])), 45, (255, 255, 255),
                       -1, cv2.LINE_AA)
            cv2.circle(overlay,
                       (int(single_trajectory.iloc[0].loc['x_center']), int(single_trajectory.iloc[0].loc['y_center'])),
                       int(single_trajectory.iloc[0].loc['radius']), (255, 255, 255),
                       1, cv2.LINE_AA)

            alpha = 0.6

            pad = cv2.addWeighted(overlay, alpha, pad, 1 - alpha, 0)

            cv2.putText(pad, '{}'.format(data_to_plot), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))

            pad_copy = pad.copy()
        print(single_trajectory.iloc[0].loc['radius'])
        pad = pad_copy.copy()
        text_pad = empty_text.copy()
        cv2.polylines(pad, [np.array([[int(single_trajectory.iloc[0].loc['x_center']), int(single_trajectory.iloc[0].loc['y_center'])],
                                      [single_trajectory.loc[frame_index, 'point1'][0], single_trajectory.loc[frame_index, 'point1'][1]]])],
                      False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(pad, (single_trajectory.loc[frame_index, 'point1'][0], single_trajectory.loc[frame_index, 'point1'][1]), 3, (255, 255, 255),
                   -1, cv2.LINE_AA)
        text_ls = ['distance', 'angle', 'bearing', 'speed', 'length', 'time', 'start_time', 'turn_rate', 'conc_s1',
                   'd_conc_s1', 'condition', 'trajectory_index', 'point1', 'point2']
        for text_ind, text in enumerate(text_ls):
            content = single_trajectory.loc[frame_index, text]
            if isinstance(content, Number) and text not in ['time', 'start_time', 'turn_rate', 'conc_s1', 'd_conc_s1',
                                                            'point1', 'point2']:
                cv2.putText(text_pad, '{}: {}'.format(text.title(), np.round(content, 1)), (10, 30 + 50*text_ind),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
            else:
                cv2.putText(text_pad, '{}: {}'.format(text.title(), content), (10, 30 + 50 * text_ind),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.imshow('draw_pad', pad)
        cv2.imshow('text_board', text_pad)
        input = cv2.waitKeyEx(-1)

        if input == 2424832:  # Left Arrow Key
            # frame_index -= 1
            # if frame_index < 0:
            #     frame_index = 0
            curr_trajectory_index -= 1
            if curr_trajectory_index < 0:
                curr_trajectory_index = 0
        elif input == 2555904:  # Right Arrow Key
            # frame_index += 1
            # if frame_index >= lines.shape[0]:
            #     frame_index = lines.shape[0] - 1
            curr_trajectory_index += 1
            if frame_index >= lines['trajectory_index'].max():
                frame_index = lines['trajectory_index'].max()
        elif input == 27:  # Esc Key
            cv2.destroyAllWindows()
            break

def screen2lines(lines,screen):
    out_df = pd.DataFrame()
    condition = -1
    trajectory_index = -1
    for row_num in range(screen.shape[0]):
        current_condition = screen.iloc[row_num].loc['condition']
        if current_condition != condition:
            condition_lines = condition_filter(lines, current_condition)
            condition = current_condition
        current_trajectory_index = screen.iloc[row_num].loc['trajectory_index']
        if current_trajectory_index != trajectory_index:
            single_trajectory = condition_lines[condition_lines['trajectory_index'] == current_trajectory_index]
            trajectory_index = current_trajectory_index
        temp_lines = single_trajectory[(single_trajectory['start_time'] >= screen.iloc[row_num].loc['start_time']) & (single_trajectory['start_time'] <= screen.iloc[row_num].loc['end_time'])]
        out_df = out_df.append(temp_lines, ignore_index=True)
    return out_df

if __name__ == '__main__':
    # # # sample = pd.read_pickle('selected_trajectories\\12hr_13.pkl')
    # sample = pd.read_pickle('trajectories\\36hr_29.pkl')
    # pts = trajectory2pts(sample)
    # circle_fit(pts)
    # exit()
    # lines = line_generator(sample, pts, '12hr_7')
    # # lines = lines[pd.notnull(lines['angle'])]
    # # lines = lines[pd.notnull(lines['distance'])]
    # lines = lines[lines['on_edge'] == 0]
    # # lines = lines[lines['start_time'] < 30]
    # lines = lines.reset_index(drop=True)
    # # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # #     print(lines)
    # draw_single_trajectory(lines)
    # exit()
    # pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    # pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
    # cv2.polylines(pad, [pts], color=(255, 255, 255), isClosed=False, thickness=1, lineType=cv2.LINE_AA)
    # yc, xc, R, residu, area = circle_fit(pts)
    # cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
    #            int(3), color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    # cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
    #            np.round((R - edge_size * scale), 0).astype(int), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    # cv2.circle(pad, (np.round(xc, 0).astype(int), np.round(yc, 0).astype(int)),
    #            np.round(R, 0).astype(int), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    #
    # cv2.imshow('test1', pad)
    # cv2.waitKey(-1)
    #
    # exit()

    # *********************************************************************************************************************


    # directory = os.fsencode('trajectories')
    #
    # for file in os.listdir(directory):
    #      filename = os.fsdecode(file)
    #      if filename.endswith(".pkl"):
    #          print(filename)
    #          sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
    #
    #          if circle_fit_selection(sample):
    #              sample.to_pickle('selected_trajectories\\{}'.format(filename))
    #
    # exit()




    # *********************************************************************************************************************
    # names = ['Fed','12hr','24hr','48hr','60hr','72hr']
    # box_data = []
    #
    # for name in names:
    #     lines = pd.read_pickle('{}_lines.pkl'.format(name))
    #
    #     lines = lines[pd.notnull(lines['angle'])]
    #     # lines = lines[pd.notnull(lines['distance'])]
    #     lines = lines[lines['on_edge'] == 0]
    #     lines = lines[lines['time'] == 0.5]
    #     lines = lines.reset_index(drop=True)
    #
    #     temp_boxdata = []
    #     for trajectory_name in lines['trajectory_name'].unique():
    #         trajectory_lines = lines[lines['trajectory_name'] == trajectory_name]
    #         temp_boxdata.append(trajectory_lines[trajectory_lines['angle']>=30].shape[0]/trajectory_lines.shape[0])
    #
    #     box_data.append(temp_boxdata)
    #
    # sns.boxplot(data=box_data)
    # plt.show()
    # exit()


    # *********************************************************************************************************************
    # total_lines = line_template
    #
    # directory = os.fsencode('selected_trajectories')
    #
    # # for condition in conditions:
    # for file in os.listdir(directory):
    #
    #     filename = os.fsdecode(file)
    #     print(filename)
    #     if filename.endswith(".pkl"):
    #         sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
    #         total_lines = total_lines.append(line_generator(sample, trajectory2pts(sample), filename.split('.')[0]),
    #                                          ignore_index=True)
    #
    # total_lines.to_pickle('all_selected_lines.pkl')
    # total_lines.to_pickle('all_selected_lines_bz2.pkl', compression='bz2')

    # ******************************************************************************************************************
    # a = pd.read_pickle('all_selected_lines.pkl')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(a)
    # exit()
    total_lines = line_template
    todo_ls = []
    directory = os.fsencode('trajectories')

    # for condition in conditions:
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # finished_index = [10,1,101,102,104,105,107,11,111,110,113,114,116,117,106,0,118,12,120,124,125,127,128,13,130,131,129,132,133,134,119,138,136,103,14,112,141,142,115,144,146,145,147,135,148,150,153,109,123,154,155,143,137,157,158,159,16,160,163,126,162,164,100,167,17,108,171,172,166,156,174,149,175,139,176,121,177,178,18,180,182,184,169,161,186,168,189,151,165,19,122,190,140,193,194,183,15,196,192,197,2,201,202,203,204,170,152,205,188,206,195,209,179,210,211,212,213,215,216,217,218,220,207,23,208,24,26,219,214,28,173,30,187,31,21,191,199,32,22,38,37,3,41,181,42,200,198,20,185,48,36,35,49,25,50,27,43,51]
        # if int(filename.split('.')[0].split('_')[1]) in finished_index:
        #     continue
        # if filename.endswith(".pkl") and filename.startswith("36hr"):
        if filename.endswith(".pkl"):
            sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
            todo_ls.append([sample, filename])

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    results = Parallel(n_jobs=num_cores)(
        delayed(line_generator)(sample, trajectory2pts(sample), filename.split('.')[0]) for [sample, filename] in
        todo_ls)
    for result in results:
        total_lines = total_lines.append(result, ignore_index=True)
    total_lines.to_pickle('all_lines.pkl')

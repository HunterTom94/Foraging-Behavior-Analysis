import numpy as np
import cv2
import pandas as pd
from scipy import optimize, stats
import os
import itertools
import math
from geometry import signed_angle_between, angle_between
from opencv_drawing import LinkPoints, drawPolyline
from matplotlib import cm
from time import time
from data_management import condition_filter
from joblib import Parallel, delayed
import multiprocessing

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

scale = 10
edge_size = 2.5
line_template = pd.DataFrame(
    columns=['line_ID', 'point1', 'point2', 'angle', 'length', 'speed', 'distance', 'time', 'start_time',
             'on_edge', 'condition', 'trajectory_index'])
gaus_x = np.linspace(-100, 100, 20000)
thres_ind = np.argwhere(gaussian(gaus_x,0,0.5)> 0.0001)[0]
assert thres_ind > 0
gaus_thres_sig0p5 = gaus_x[thres_ind]


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
    if area > 260000 and area < 280000 and np.sqrt(residu) < 200:
        return True
    else:
        return False


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

    diff_time = np.diff(trajectory.iloc[:, 2])

    yc, xc, R, _, _ = circle_fit(pts)

    lines = line_template
    skip_angle = 0
    for ind, (pt1, pt2) in enumerate(pairwise(pts)):
        # Calculate Length Travelled
        length = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        # Calculate turn angle
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
                    skip_angle = 1
                    break
            if not skip_angle:
                # angle = signed_angle_between(pt2 - pt1, lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1'])
                # Negative Sign on y coordinate to be in Accordance with draw trajectory (x-y axis swap)
                angle = signed_angle_between((lines.loc[last_line, 'point2'] - lines.loc[last_line, 'point1']) * np.array([1,-1]), (pt2 - pt1) * np.array([1,-1]))
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
        # Calculate Bearing
        # Negative Sign on y coordinate to be in Accordance with draw trajectory (x-y axis swap)
        bearing = signed_angle_between((pt2 - pt1) * np.array([1,-1]), (np.array([xc, yc]) - pt1) * np.array([1,-1]))
        # Calculating concentration
        conc_s0p5 = gauss_conc(dist, 0.5)[0]
        conc_s1 = gauss_conc(dist, 1)[0]
        conc_s2 = gauss_conc(dist, 2)[0]
        conc_s3 = gauss_conc(dist, 3)[0]

        lines = lines.append(
            {'line_ID': ind, 'point1': pt1, 'point2': pt2, 'angle': np.round(angle, 1),
             'turn_rate': np.round(angle / dist, 3),
             'turn_speed': np.round(angle / diff_time[ind], 1), 'length': np.round(length, 1),
             'speed': np.round(length / diff_time[ind], 1), 'distance': dist,
             'time': diff_time[ind], 'start_time': trajectory.iloc[ind, 2], 'on_edge': on_edge,
             'condition': trajectory_name.split('_')[0], 'trajectory_index': int(trajectory_name.split('_')[1]),
             'bearing': np.round(bearing, 1), 'conc_s0p5': conc_s0p5, 'conc_s1': conc_s1, 'conc_s2': conc_s2,
             'conc_s3': conc_s3},
            ignore_index=True)
    lines['delta_distance'] = np.round(np.insert(np.diff(lines['distance'].values), 0, np.nan), 1)
    lines['d_dist_rate'] = np.insert(np.round((np.diff(lines['distance'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s0p5'] = np.insert(np.round((np.diff(lines['conc_s0p5'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s1'] = np.insert(np.round((np.diff(lines['conc_s1'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s2'] = np.insert(np.round((np.diff(lines['conc_s2'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
    lines['d_conc_s3'] = np.insert(np.round((np.diff(lines['conc_s3'].values) / lines['time'].values[:-1]).astype('float64'),5), 0, np.nan)
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
    text_pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    pad = np.zeros((70 * scale, 70 * scale), dtype=np.uint8)
    pad = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
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
    cv2.moveWindow("draw_pad", 20, 20)
    cv2.namedWindow("text_board")
    cv2.moveWindow("text_board", 520, 1020)

    while True:
        curr_trajectory_index = lines.loc[frame_index, 'trajectory_index']
        if curr_trajectory_index != trajectory_index:
            pad = empty_pad.copy()
            trajectory_index = curr_trajectory_index
            single_trajectory = lines[lines['trajectory_index'] == trajectory_index]
            inds = np.digitize(single_trajectory[data_to_plot].values / (single_trajectory[data_to_plot].max() / 255), np.arange(256)) - 1
            for index, (_, row) in enumerate(single_trajectory.iterrows()):
                LinkPoints(pad, row['point1'], row['point2'], BGR=tuple(color_ls[inds[index]].tolist()))

            overlay = pad.copy()

            cv2.circle(overlay, (int(pad.shape[0] / 2), int(pad.shape[1] / 2)), 45, (255, 255, 255),
                       -1, cv2.LINE_AA)

            alpha = 0.6

            pad = cv2.addWeighted(overlay, alpha, pad, 1 - alpha, 0)

            pad_copy = pad.copy()

        pad = pad_copy.copy()

        cv2.circle(pad, (lines.loc[frame_index, 'point1'][0], lines.loc[frame_index, 'point1'][1]), 3, (255, 255, 255),
                   -1, cv2.LINE_AA)
        cv2.putText(pad, 'Distance: {}'.format(np.round(lines.loc[frame_index, 'distance']), 1), (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0))
        cv2.putText(pad, 'TimeStamp: {}'.format(np.round(lines.loc[frame_index, 'start_time']), 1), (400, 680),
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
        out_df = out_df.append(temp_lines,ignore_index=True)
    return out_df

if __name__ == '__main__':
    # sample = pd.read_pickle('selected_trajectories\\12hr_94.pkl')
    # pts = trajectory2pts(sample)
    # lines = line_generator(sample, pts, 'Fed_9')
    # # lines = lines[pd.notnull(lines['angle'])]
    # # lines = lines[pd.notnull(lines['distance'])]
    # lines = lines[lines['on_edge'] == 0]
    # lines = lines.reset_index(drop=True)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(lines)
    # # draw_trajectory(lines)
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
    a = pd.read_pickle('all_selected_lines.pkl')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(a)
    exit()
    total_lines = line_template
    todo_ls = []
    directory = os.fsencode('selected_trajectories')

    # for condition in conditions:
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pkl"):
            sample = pd.read_pickle(os.path.join(os.fsdecode(directory), filename))
            todo_ls.append([sample, filename])

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(line_generator)(sample, trajectory2pts(sample), filename.split('.')[0]) for [sample, filename] in
        todo_ls)
    for result in results:
        total_lines = total_lines.append(result, ignore_index=True)
    # total_lines.to_pickle('all_selected_lines.pkl')
    total_lines.to_pickle('all_selected_lines.bz2')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_management import condition_filter, just_before_found, meta_update
from foraging_util import draw_trajectory, draw_single_trajectory,  screen2lines
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import multiprocessing
import math
from scipy import stats
from scipy.integrate import trapz
from time import time
from scipy.optimize import curve_fit
from scipy.stats import norm

data = pd.read_pickle('all_selected_lines.pkl')
meta = pd.read_pickle('metadata.pkl')

scale = 10
slow_down_corrcoef_threshold = -0.7
dist_corrcoef_threshold = 0.4
conc_corrcoef_threshold = -0.4
bear_corrcoef_threshold = -0.4
orthokinesis_duration = 3
weathervane_duration = 3
curve_duration = 3
sharp_turn_duration = 1.5
angle_threshold = 90
length_threshold = 6*scale
conditions = ['Fed','12hr','24hr','36hr','48hr','60hr','72hr']

num_cores = multiprocessing.cpu_count()


def orthokinesis_screen(lines, duration):
    def find_overlap(lines, df):
        df['bout_index'] = np.nan
        continuous_bout_index = 0
        for row_num in range(df.shape[0]):
            df.at[row_num, 'bout_index'] = continuous_bout_index
            if row_num < df.shape[0] - 1:
                if not (df.iloc[row_num].loc['end_time'] >= df.iloc[row_num + 1].loc['start_time'] and
                        df.iloc[row_num].loc['end_time'] <= df.iloc[row_num + 1].loc['end_time']):
                    continuous_bout_index += 1
        new_df = pd.DataFrame()
        for bout_index in np.unique(df['bout_index']):
            bout_df = df[df['bout_index'] == bout_index]
            corr_rows = lines[(lines['start_time'] >= bout_df.iloc[0].loc['start_time']) & (
                        lines['start_time'] <= bout_df.iloc[-1].loc['end_time'])]
            x = corr_rows['distance'].values
            y = corr_rows['speed'].values
            z = corr_rows['start_time'].values
            corrcoef = pearsonr(x, y)
            slow_down_corrcoef = pearsonr(y, z)
            if corrcoef[1] < 0.05 and corrcoef[0] > dist_corrcoef_threshold and slow_down_corrcoef[1] < 0.05 and slow_down_corrcoef[0] < slow_down_corrcoef_threshold and corr_rows['distance'].min() < 8*scale:
                new_df = new_df.append(
                    {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                     'start_time': corr_rows.iloc[0].loc['start_time'],
                     'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)

        return new_df

    out_df = pd.DataFrame()
    lines = lines[lines['speed'] != 0]
    for condition in conditions:
        lines_condition = condition_filter(lines, condition)
        for trajectory_index in np.unique(lines_condition['trajectory_index']):
            print(trajectory_index)
            trajectory_df = pd.DataFrame()
            single_trajectory = lines_condition[lines_condition['trajectory_index'] == trajectory_index]
            for row_index in range(single_trajectory.shape[0]):
                corr_rows = single_trajectory[
                    (single_trajectory['start_time'] >= single_trajectory.iloc[row_index].loc['start_time']) & (
                                single_trajectory['start_time'] < single_trajectory.iloc[row_index].loc[
                            'start_time'] + duration)]
                if corr_rows.shape[0] >= 3:
                    x = corr_rows['distance'].values
                    y = corr_rows['speed'].values
                    z = corr_rows['start_time'].values
                    corrcoef = pearsonr(x, y)
                    slow_down_corrcoef = pearsonr(y, z)
                    if corrcoef[1] < 0.05 and corrcoef[0] > dist_corrcoef_threshold and slow_down_corrcoef[1] < 0.05 and slow_down_corrcoef[0] < slow_down_corrcoef_threshold:
                        trajectory_df = trajectory_df.append(
                            {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                             'start_time': corr_rows.iloc[0].loc['start_time'],
                             'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)
            out_df = out_df.append(find_overlap(single_trajectory, trajectory_df), ignore_index=True)
    return out_df

def orthokinesis_screen_conc(lines, duration,conc_label):
    def find_overlap(lines, df):
        df['bout_index'] = np.nan
        continuous_bout_index = 0
        for row_num in range(df.shape[0]):
            df.at[row_num, 'bout_index'] = continuous_bout_index
            if row_num < df.shape[0] - 1:
                if not (df.iloc[row_num].loc['end_time'] >= df.iloc[row_num + 1].loc['start_time'] and
                        df.iloc[row_num].loc['end_time'] <= df.iloc[row_num + 1].loc['end_time']):
                    continuous_bout_index += 1
        new_df = pd.DataFrame()
        for bout_index in np.unique(df['bout_index']):
            bout_df = df[df['bout_index'] == bout_index]
            corr_rows = lines[(lines['start_time'] >= bout_df.iloc[0].loc['start_time']) & (
                        lines['start_time'] <= bout_df.iloc[-1].loc['end_time'])]
            x = corr_rows[conc_label].values
            y = corr_rows['speed'].values
            z = corr_rows['start_time'].values
            corrcoef = pearsonr(x, y)
            slow_down_corrcoef = pearsonr(y, z)
            if corrcoef[1] < 0.05 and corrcoef[0] < conc_corrcoef_threshold and slow_down_corrcoef[1] < 0.05 and slow_down_corrcoef[0] < slow_down_corrcoef_threshold and corr_rows['distance'].min() < 8*scale:
                new_df = new_df.append(
                    {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                     'start_time': corr_rows.iloc[0].loc['start_time'],
                     'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)

        return new_df

    out_df = pd.DataFrame()
    lines = lines[lines['speed'] != 0]
    for condition in conditions:
        lines_condition = condition_filter(lines, condition)
        for trajectory_index in np.unique(lines_condition['trajectory_index']):
            print(trajectory_index)
            trajectory_df = pd.DataFrame()
            single_trajectory = lines_condition[lines_condition['trajectory_index'] == trajectory_index]
            for row_index in range(single_trajectory.shape[0]):
                corr_rows = single_trajectory[
                    (single_trajectory['start_time'] >= single_trajectory.iloc[row_index].loc['start_time']) & (
                                single_trajectory['start_time'] < single_trajectory.iloc[row_index].loc[
                            'start_time'] + duration)]
                if corr_rows.shape[0] >= 3:
                    x = corr_rows[conc_label].values
                    y = corr_rows['speed'].values
                    z = corr_rows['start_time'].values
                    corrcoef = pearsonr(x, y)
                    slow_down_corrcoef = pearsonr(y, z)
                    if corrcoef[1] < 0.05 and corrcoef[0] < conc_corrcoef_threshold and slow_down_corrcoef[1] < 0.05 and slow_down_corrcoef[0] < slow_down_corrcoef_threshold:
                        trajectory_df = trajectory_df.append(
                            {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                             'start_time': corr_rows.iloc[0].loc['start_time'],
                             'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)
            out_df = out_df.append(find_overlap(single_trajectory, trajectory_df), ignore_index=True)
    return out_df

def weathervane_screen(lines, duration):
    def find_overlap(lines, df):
        df['bout_index'] = np.nan
        continuous_bout_index = 0
        for row_num in range(df.shape[0]):
            df.at[row_num, 'bout_index'] = continuous_bout_index
            if row_num < df.shape[0] - 1:
                if not (df.iloc[row_num].loc['end_time'] >= df.iloc[row_num + 1].loc['start_time'] and
                        df.iloc[row_num].loc['end_time'] <= df.iloc[row_num + 1].loc['end_time']):
                    continuous_bout_index += 1
        new_df = pd.DataFrame()
        for bout_index in np.unique(df['bout_index']):
            bout_df = df[df['bout_index'] == bout_index]
            corr_rows = lines[(lines['start_time'] >= bout_df.iloc[0].loc['start_time']) & (
                        lines['start_time'] <= bout_df.iloc[-1].loc['end_time'])]
            curve_test_angle = corr_rows['angle'].values[1:]
            curve_test_bearing = corr_rows['bearing'].values[:-1]
            if not all(x > 0 for x in curve_test_angle * curve_test_bearing):
                continue
            x = np.abs(corr_rows['bearing'].values)
            y = corr_rows['start_time'].values
            corrcoef = pearsonr(x, y)
            if corrcoef[1] < 0.05 and corrcoef[0] < bear_corrcoef_threshold and corr_rows['distance'].min() < 20*scale:
                new_df = new_df.append(
                    {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                     'start_time': corr_rows.iloc[0].loc['start_time'],
                     'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)

        return new_df

    out_df = pd.DataFrame()
    lines = lines[lines['speed'] != 0]
    lines = lines[lines['on_edge'] != 1]
    for condition in conditions:
        lines_condition = condition_filter(lines, condition)
        for trajectory_index in np.unique(lines_condition['trajectory_index']):
            print(trajectory_index)
            trajectory_df = pd.DataFrame()
            single_trajectory = lines_condition[lines_condition['trajectory_index'] == trajectory_index]
            for row_index in range(single_trajectory.shape[0]):
                corr_rows = single_trajectory[
                    (single_trajectory['start_time'] >= single_trajectory.iloc[row_index].loc['start_time']) & (
                                single_trajectory['start_time'] < single_trajectory.iloc[row_index].loc[
                            'start_time'] + duration)]
                if corr_rows.shape[0] >= 3:
                    curve_test_angle = corr_rows['angle'].values[1:]
                    curve_test_bearing = corr_rows['bearing'].values[:-1]
                    if not all(x > 0 for x in curve_test_angle * curve_test_bearing):
                        continue

                    x = np.abs(corr_rows['bearing'].values)
                    y = corr_rows['start_time'].values
                    corrcoef = pearsonr(x, y)
                    if corrcoef[1] < 0.05 and corrcoef[0] < bear_corrcoef_threshold:
                        trajectory_df = trajectory_df.append(
                            {'condition': condition, 'trajectory_index': trajectory_index, 'corrcoef': corrcoef[0],
                             'start_time': corr_rows.iloc[0].loc['start_time'],
                             'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)
            out_df = out_df.append(find_overlap(single_trajectory, trajectory_df), ignore_index=True)
    return out_df

def curve_screen(lines, duration):
    def find_overlap(lines, df):
        df['bout_index'] = np.nan
        continuous_bout_index = 0
        for row_num in range(df.shape[0]):
            df.at[row_num, 'bout_index'] = continuous_bout_index
            if row_num < df.shape[0] - 1:
                if not (df.iloc[row_num].loc['end_time'] >= df.iloc[row_num + 1].loc['start_time'] and
                        df.iloc[row_num].loc['end_time'] <= df.iloc[row_num + 1].loc['end_time']):
                    continuous_bout_index += 1
        new_df = pd.DataFrame()
        for bout_index in np.unique(df['bout_index']):
            bout_df = df[df['bout_index'] == bout_index]
            corr_rows = lines[(lines['start_time'] >= bout_df.iloc[0].loc['start_time']) & (
                        lines['start_time'] <= bout_df.iloc[-1].loc['end_time'])]
            curve_test_angle = corr_rows['angle'].values[1:]
            curve_test_bearing = corr_rows['bearing'].values[:-1]
            if not all(x > 0 for x in curve_test_angle * curve_test_bearing):
                continue
            new_df = new_df.append(
                {'condition': condition, 'trajectory_index': trajectory_index,
                 'start_time': corr_rows.iloc[0].loc['start_time'],
                 'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)

        return new_df

    out_df = pd.DataFrame()
    lines = lines[lines['speed'] != 0]
    lines = lines[lines['on_edge'] != 1]
    for condition in conditions:
        lines_condition = condition_filter(lines, condition)
        for trajectory_index in np.unique(lines_condition['trajectory_index']):
            print(trajectory_index)
            trajectory_df = pd.DataFrame()
            single_trajectory = lines_condition[lines_condition['trajectory_index'] == trajectory_index]
            for row_index in range(single_trajectory.shape[0]):
                corr_rows = single_trajectory[
                    (single_trajectory['start_time'] >= single_trajectory.iloc[row_index].loc['start_time']) & (
                                single_trajectory['start_time'] < single_trajectory.iloc[row_index].loc[
                            'start_time'] + duration)]
                if corr_rows.shape[0] >= 3:
                    curve_test_angle = corr_rows['angle'].values[1:]
                    curve_test_bearing = corr_rows['bearing'].values[:-1]
                    if not all(x > 0 for x in curve_test_angle * curve_test_bearing):
                        continue

                    trajectory_df = trajectory_df.append(
                        {'condition': condition, 'trajectory_index': trajectory_index,
                         'start_time': corr_rows.iloc[0].loc['start_time'],
                         'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)
            out_df = out_df.append(find_overlap(single_trajectory, trajectory_df), ignore_index=True)
    return out_df

def sharp_turn_screen(lines, duration):
    def find_overlap(lines, df, trajectory_index):
        df['bout_index'] = np.nan
        continuous_bout_index = 0
        for row_num in range(df.shape[0]):
            df.at[row_num, 'bout_index'] = continuous_bout_index
            if row_num < df.shape[0] - 1:
                if not (df.iloc[row_num].loc['end_time'] >= df.iloc[row_num + 1].loc['start_time'] and
                        df.iloc[row_num].loc['end_time'] <= df.iloc[row_num + 1].loc['end_time']):
                    continuous_bout_index += 1
        new_df = pd.DataFrame()
        for bout_index in np.unique(df['bout_index']):
            bout_df = df[df['bout_index'] == bout_index]
            corr_rows = lines[(lines['start_time'] >= bout_df.iloc[0].loc['start_time']) & (
                    lines['start_time'] <= bout_df.iloc[-1].loc['end_time'])]
            center = np.mean(corr_rows['point1'].values)
            dist = math.hypot(center[0] - corr_rows.iloc[0].loc['x_center'], center[1] - corr_rows.iloc[0].loc['y_center'])
            if_found = int(np.isnan(meta[(meta['condition'] == condition) & (meta['trajectory_index'] == trajectory_index)]['find_time']))
            found_ls = ['found', 'not_found']
            turn_sum = int(np.nansum(corr_rows['angle'].values))
            turn = (180 + turn_sum) % 360 - 180
            new_df = new_df.append(
                {'condition': condition, 'trajectory_index': trajectory_index,
                 'start_time': corr_rows.iloc[0].loc['start_time'],
                 'end_time': corr_rows.iloc[-1].loc['start_time'], 'distance': dist, 'found': found_ls[if_found], 'angle_sum': turn_sum, 'angle': turn}, ignore_index=True)

        return new_df

    def screen_trajectory(trajectory_index):
        print(trajectory_index)
        trajectory_df = pd.DataFrame()
        single_trajectory = lines_condition[lines_condition['trajectory_index'] == trajectory_index]
        for row_index in range(single_trajectory.shape[0]):
            corr_rows = single_trajectory[
                (single_trajectory['start_time'] >= single_trajectory.iloc[row_index].loc['start_time']) & (
                        single_trajectory['start_time'] < single_trajectory.iloc[row_index].loc[
                    'start_time'] + duration)]
            if corr_rows.shape[0] >= 3:
                if -180 in corr_rows['angle'].values or 180 in corr_rows['angle'].values:
                    continue
                turn_sum = np.nansum(corr_rows['angle'].values)
                length_sum = np.nansum(corr_rows['length'].values)
                if length_sum < 2 * scale:
                    continue
                if np.abs(turn_sum) > angle_threshold and length_sum < length_threshold:
                    trajectory_df = trajectory_df.append(
                        {'condition': condition, 'trajectory_index': trajectory_index,
                         'start_time': corr_rows.iloc[0].loc['start_time'],
                         'end_time': corr_rows.iloc[-1].loc['start_time']}, ignore_index=True)
        return find_overlap(single_trajectory, trajectory_df, trajectory_index)

    out_df = pd.DataFrame()
    lines = lines[lines['on_edge'] != 1]
    lines = lines[pd.notnull(lines['angle'])]
    # lines = lines[lines['speed'] != 0]
    for condition in conditions:
        lines_condition = condition_filter(lines, condition)

        results = Parallel(n_jobs=num_cores)(
            delayed(screen_trajectory)(trajectory_index) for trajectory_index in
            np.unique(lines_condition['trajectory_index']))
        for result in results:
            out_df = out_df.append(result, ignore_index=True)
    return out_df

def kde_gen(x):
    def func(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    start_time = time()
    below_zero = True
    fit_range = 2
    bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
    support = np.linspace(np.min(x), np.max(x), 1000)

    kernels = []
    count = 0
    length = len(x)
    for x_i in x:
        print(count/length)
        kernel = stats.norm(x_i, bandwidth).pdf(support)
        kernels.append(kernel)
        count += 1

    density = np.sum(kernels, axis=0)
    density /= trapz(density, support)
    peak_index = np.argmax(density)

    while below_zero:
        peak_range_index = support[int(len(support) - (len(support)-peak_index)*fit_range) : (len(support) - 1)]
        peak_range = density[int(len(support) - (len(support)-peak_index)*fit_range) : (len(support) - 1)]

        popt, pcov = curve_fit(func, peak_range_index, peak_range, p0=[np.max(density), support[peak_index], 0.5])
        pred = func(support, popt[0], popt[1], popt[2])

        cleaned = density - pred

        below_zero = np.any(cleaned[support < support[peak_index]] < 0)
        fit_range -= 0.05
        if fit_range < 1:
            break

    np.save('{}_distribution_no_tail.npy'.format(condition), np.stack((support,cleaned, density)))

    plt.plot(support, pred, c='r')
    plt.plot(support, density, c='k')
    plt.plot(support, cleaned, c='b')

    plt.xlabel('distance (millimeter * 10)')
    plt.ylabel('density')
    plt.title(condition)
    plt.ylim([0, 0.02])
    plt.xlim([0, 300])

    plt.show()

    # print()
    # print(fit_range)
    # print('time spent {}'.format(time()-start_time))

def fit_head(kde, condition):
    def func(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    support = kde[0, :]
    cleaned = kde[1, :]
    density = kde[2, :]
    cleaned /= trapz(cleaned, support)

    fit_range = 0
    error = 0
    error_threshold = 0.00001

    while error < error_threshold:
        fit_range += 1
        peak_range_index = support[support < fit_range]
        peak_range = cleaned[support < fit_range]

        popt, pcov = curve_fit(func, peak_range_index, peak_range, p0=[np.max(peak_range), peak_range_index[-1], 0.5])
        pred = func(support, popt[0], popt[1], popt[2])
        error = np.abs(pred[support < fit_range] - peak_range).sum() / len(peak_range_index)

    ax = plt.gca()
    plt.plot(support, pred, c='r')
    plt.plot(support, cleaned, c='b')

    plt.text(0.05, 0.95, 'error < {}'.format(error_threshold), transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

    # Plot Gaussian Height
    x1= popt[1]
    y, h, col = 0, popt[0], 'k'
    plt.plot([x1, x1], [y, y + h], lw=1.5, c=col)
    plt.text(x1 + 1, popt[0] * .5,  str(np.round(x1,1)), ha='left', va='center', color=col)

    # Plot Std
    x1, x2 = popt[1],popt[1] - popt[2]
    y, h, col = popt[0] + 0.0002, 0.0002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, str(np.round(popt[2],1)), ha='center', va='bottom', color=col)



    plt.xlabel('distance (millimeter * 10)')
    plt.ylabel('density')
    plt.title(condition)
    plt.ylim([0, 0.01])
    plt.xlim([0, 300])
    plt.savefig('new_data\\{}_head_fit_error_{}'.format(condition, error_threshold))
    plt.show()

def head_area(kde, area_till):
    support = kde[0, :]
    cleaned = kde[1, :]
    density = kde[2, :]
    cleaned /= trapz(cleaned, support)

    till_index = np.argmin(np.abs(support - area_till))


    # Plot Area
    x1 = area_till
    y, h, col = 0, cleaned[till_index], 'g'
    plt.plot([x1, x1], [y, y + h], lw=1.5, c=col, linestyle='dashed')
    plt.text(x1 - 1, cleaned[till_index] * .5, str(np.round(trapz(cleaned[:till_index], support[:till_index]), 3)), ha='right', va='center', color='k')
    plt.plot(support, cleaned, c='b')

    plt.xlabel('distance (millimeter * 10)')
    plt.ylabel('density')
    plt.title(condition)
    plt.ylim([0, 0.01])
    plt.xlim([0, 300])
    plt.savefig('new_data\\{}_head_area_till_{}'.format(condition, area_till))
    plt.show()

if __name__ == '__main__':
    # found_before = pd.read_pickle('selected_found_before.pkl')  # Includes not found
    # found_after = pd.read_pickle('selected_found_after.pkl')
    # found = pd.read_pickle('selected_found.pkl')
    # not_found = pd.read_pickle('selected_not_found.pkl')
    # meta_v1 = pd.read_pickle('metadata_v1.pkl')
    # meta_v2 = pd.read_pickle('metadata_v2.pkl')
    # meta_v2 = meta_v2[meta_v2['selected'] == 1]
    # orthokinesis = pd.read_pickle('orthokinesis_dur_3_slow_-0.7_dis_0.4.pkl')
    # conc_orthokinesis = pd.read_pickle('orthokinesis_dur_3_slow_-0.7_dis_0.4_conc_conc_s3.pkl')
    # potential_weathervane = pd.read_pickle('weathervane_dur_3_bear_-0.4.pkl')
    # potential_curve = pd.read_pickle('curve_dur_3.pkl')
    # sharp_turn = pd.read_pickle('sharp_turn_dur_1.5_angle_90.pkl')
    # noafter_sharp_turn = pd.read_pickle('noafter_sharp_turn_dur_1.5_angle_90.pkl')
    # after_sharp_turn = pd.read_pickle('after_sharp_turn_dur_1.5_angle_90.pkl')
    #
    # test_72 = pd.read_pickle('all_lines_test_72hr.pkl')

    # all_lines = pd.read_pickle('all_lines.pkl')
    # all_lines = all_lines[all_lines['radius'] < 305]
    # all_lines = all_lines[all_lines['distance'] < 320]

    # draw_single_trajectory(condition_filter(all_lines, '72hr'))
    # exit()

    # *********************************************************************************************************************
    # all_groupby = all_lines.groupby(['condition', 'trajectory_index'])
    # radius = all_groupby['radius'].mean().values
    # g = sns.distplot(radius, kde=False, hist_kws=dict(edgecolor="w", linewidth=1), bins=5)
    # g.set(xlabel='millimeter * 10', ylabel='count')
    # plt.show()

    # exit()
    # *********************************************************************************************************************
    for condition in conditions:
        kde = np.load('{}_distribution_no_tail.npy'.format(condition))
        # fit_head(kde, condition)
        head_area(kde, 150)
    exit()
    # *********************************************************************************************************************
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(kde_gen)(condition_filter(all_lines, condition)['distance'] for condition in conditions))

    for condition in conditions:
        condition_line = condition_filter(all_lines, condition)
        kde_gen(condition_line['distance'])
        # plt.ylim([0, 0.02])
        # plt.xlim([0, 300])
        #
        # plt.show()
        # exit()
        # g = sns.distplot(condition_line['distance'], norm_hist=True)
        # g.set(xlabel='distance (millimeter * 10)', ylabel='density', title=condition)
        # plt.ylim([0, 0.02])
        # plt.xlim([0, 325])
        # # plt.legend(conditions)
        # plt.show()
        # g = sns.distplot(condition_line['distance'], hist=False, rug=True, rug_kws={'alpha': 0})
        # # g = sns.distplot(condition_line['distance'], kde=False, hist_kws=dict(edgecolor="w", linewidth=1), norm_hist=True)
        # g.set(xlabel='distance (millimeter * 10)', ylabel='density', title=condition)
        # plt.ylim([0, 0.02])
        # plt.xlim([0, 325])
    # plt.legend(conditions)
    #     plt.show()
    exit()

    # *********************************************************************************************************************

    #
    # # print(np.sort(test_72['radius'].unique()).astype(int))
    # #
    # draw_single_trajectory(condition_filter(all_lines, '72hr'))
    # exit()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta)
    # exit()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(condition_filter(found_before,'Fed')['trajectory_index'].unique())
    # exit()

    # before_time = 3
    # lines = condition_filter(just_before_found(found_before, before_time), '12hr')
    # lines = condition_filter(data, '72hr')
    # lines = condition_filter(found_before,'Fed')
    # lines = lines[lines['on_edge'] == 0]
    # lines = lines.reset_index(drop=True)
    # *********************************************************************************************************************
    potential_sharp_turn = sharp_turn_screen(data, sharp_turn_duration)
    potential_sharp_turn.to_pickle('sharp_turn_dur_{}_angle_{}.pkl'.format(sharp_turn_duration, angle_threshold))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(potential_sharp_turn)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta)
    exit()
    # meta_v2 = condition_filter(meta_v2, '72hr')
    # print(meta_v2.shape[0])
    # print(meta_v2[meta_v2['found'] == 'found'].shape[0])
    # print(meta_v2[meta_v2['found'] == 'not_found'].shape[0])
    # # # draw_single_trajectory(screen2lines(data, condition_filter(orthokinesis, 'Fed')))
    # draw_single_trajectory(screen2lines(data, condition_filter(sharp_turn, 'Fed')))
    # exit()
    # *********************************************************************************************************************
    # potential_sharp_turn = sharp_turn_screen(found_before, sharp_turn_duration)
    # potential_sharp_turn.to_pickle('noafter_sharp_turn_dur_{}_angle_{}.pkl'.format(sharp_turn_duration, angle_threshold))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(potential_sharp_turn)
    # exit()
    # *********************************************************************************************************************
    # potential_weathervane = weathervane_screen(data, 3)
    # # potential_curve = curve_screen(data, 3)
    # potential_weathervane.to_pickle('weathervane_dur_{}_bear_{}.pkl'.format(weathervane_duration, bear_corrcoef_threshold))
    # # potential_curve.to_pickle('curve_dur_{}.pkl'.format(curve_duration))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(potential_weathervane)
    # draw_single_trajectory(screen2lines(data, condition_filter(potential_weathervane, 'Fed')))
    # exit()
    # *********************************************************************************************************************
    # draw_single_trajectory(lines)
    # exit()
    # *********************************************************************************************************************
    # for conc_label in ['conc_s1','conc_s2','conc_s3']:
    #     conc_orthokinesis = orthokinesis_screen_conc(data, orthokinesis_duration, conc_label)
    # # potential_orthokinesis = orthokinesis_screen(data, orthokinesis_duration)
    #     conc_orthokinesis.to_pickle(
    #         'orthokinesis_dur_{}_slow_{}_dis_{}_conc_{}.pkl'.format(orthokinesis_duration, slow_down_corrcoef_threshold,
    #                                                         dist_corrcoef_threshold, conc_label))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(conc_orthokinesis)
    # draw_single_trajectory(screen2lines(data, condition_filter(conc_orthokinesis, 'Fed')))
    # exit()
    # *********************************************************************************************************************
    # print(np.sort(sharp_turn['angle'].unique()))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(sharp_turn[sharp_turn['distance'] >300])
    # exit()
    # *********************************************************************************************************************

    # g = sns.jointplot(x="speed", y="distance", data=lines, kind="kde", xlim=(0, 200), ylim=(0, 300))
    # # g = sns.jointplot(x="angle", y="delta_distance", data=lines, kind="kde", xlim=(-180, 180), ylim=(-100, 100))
    # g = sns.jointplot(x="angle", y="distance", data=meta_v2, kind="kde", xlim=(-180, 180), ylim=(0, 300))
    # meta_v2 = meta_v2[(meta_v2['orthokinesis'] != 0) & (meta_v2['sharp_turn'] != 0)]
    #
    # corrcoef = pearsonr(meta_v2['orthokinesis'].values, meta_v2['sharp_turn'].values)
    # print(corrcoef)
    # g = sns.jointplot(x="orthokinesis", y="sharp_turn", data=meta_v2)

    # g = (sns.jointplot(x="angle", y="distance", data=sharp_turn, color="k", xlim=(-180, 180), ylim=(0, 300)).plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    #
    # g.fig.suptitle('{}_{}'.format(lines.loc[0, 'condition'], before_time))

    # ***********************************Sharp Turn Distribution Plot********************************************************

    g = sns.distplot(noafter_sharp_turn['distance'], kde=False, hist_kws=dict(edgecolor="w", linewidth=1))
    g.set(ylabel='count')

    # *********************************************************************************************************************

    # ********************************** Orthokinesis Graph Generator **********************************************

    # meta_update(meta,conc_orthokinesis)
    # exit()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta_v2)
    # exit()


    # sns.set(style="ticks", palette="pastel")
    # # sns.barplot(x="condition", y="orthokinesis", hue="found", palette=["m", "g"], data=meta_v2, ci=68, capsize=.1)
    # sns.barplot(x="condition", y="orthokinesis", data=meta_v2, ci=68, capsize=.1)
    # # # sns.barplot(x="startvation", y="orthokinesis", data=meta_v1, ci=68, capsize=.1)
    # sns.despine(offset=10, trim=True)
    #
    # # meta_v2 = meta_v2[meta_v2['condition'] == '72hr']
    # p_value = np.round(ttest_ind(meta_v2[meta_v2['condition'] == '60hr']['orthokinesis'], meta_v2[meta_v2['condition'] == '72hr']['orthokinesis'])[1],10)
    # print(p_value)
    #
    # # x1, x2 = -0.2, 0.2
    # # y, h, col = 3.9 + 0.05, 0.05, 'k'
    # # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # # plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)
    # # #
    # x1, x2 = 4, 5
    # y, h, col = 2.8 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
    # # # #
    # x1, x2 = 3,4
    # y, h, col = 2.4 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)
    # #
    # x1, x2 = 5,6
    # y, h, col = 2.8 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
    #
    # x1, x2 = 3.8,4.2
    # y, h, col = 2.9 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
    #
    # x1, x2 = 4.8,5.2
    # y, h, col = 3.3 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)
    #
    # x1, x2 = 5.8,6.2
    # y, h, col = 3.0 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)


    # ********************************** Sharp Turn Graph Generator **********************************************

    # meta_update(meta, orthokinesis, sharp_turn)
    # exit()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta_v2)
    # exit()

    # p_value = np.round(ttest_ind(meta_v2[(meta_v2['startvation'] == 'long') & (meta_v2['found'] == 'found')]['sharp_turn'],
    #                              meta_v2[(meta_v2['startvation'] == 'long') & (meta_v2['found'] == 'not_found')]['sharp_turn'])[1], 10)
    # print(p_value)
    # exit()

    # sns.set(style="ticks", palette="pastel")
    # g = sns.barplot(x="condition", y="sharp_turn", hue="found", palette=["m", "g"], data=meta_v2, ci=68, capsize=.1)
    # # sns.barplot(x="angle", y="distance", data=sharp_turn, ci=68, capsize=.1)
    # # sns.barplot(x="condition", y="sharp_turn", data=meta_v2, ci=68, capsize=.1)
    # sns.despine(offset=10, trim=True)
    #
    # meta_v2 = meta_v2[meta_v2['condition'] == '72hr']
    # p_value = np.round(ttest_ind(meta_v2[meta_v2['found'] == 'found']['sharp_turn'],
    #                              meta_v2[meta_v2['found'] == 'not_found']['sharp_turn'])[1], 10)
    # # p_value = np.round(ttest_ind(meta_v2[meta_v2['condition'] == '72hr']['sharp_turn'],
    # #                              meta_v2[meta_v2['condition'] == '60hr']['sharp_turn'])[1], 10)
    # print(p_value)
    #
    # # plt.legend(loc=(0.05,0.9))
    #
    # x1, x2 = -0.2, 0.2
    # y, h, col = 8.1 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    # # # #
    # x1, x2 = 0.8,1.2
    # y, h, col = 5.1 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    # # # # #
    # x1, x2 = 1.8,2.2
    # y, h, col = 3.4 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "**", ha='center', va='bottom', color=col)
    # # #
    # x1, x2 = 2.8,3.2
    # y, h, col = 6.8 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    # #
    # x1, x2 = 3.8,4.2
    # y, h, col = 4.7 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    # #
    # x1, x2 = 4.8,5.2
    # y, h, col = 4 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)
    # #
    # x1, x2 = 5.8,6.2
    # y, h, col = 6.0 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col)


    # *********************************************************************************************************************

    plt.show()
    exit()

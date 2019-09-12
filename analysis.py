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
            new_df = new_df.append(
                {'condition': condition, 'trajectory_index': trajectory_index,
                 'start_time': corr_rows.iloc[0].loc['start_time'],
                 'end_time': corr_rows.iloc[-1].loc['start_time'], 'distance': dist}, ignore_index=True)

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




if __name__ == '__main__':
    found_before = pd.read_pickle('selected_found_before.pkl')
    found = pd.read_pickle('selected_found.pkl')
    not_found = pd.read_pickle('selected_not_found.pkl')
    data = pd.read_pickle('all_selected_lines.pkl')
    meta = pd.read_pickle('metadata.pkl')
    meta_v1 = pd.read_pickle('metadata_v1.pkl')
    meta_v2 = pd.read_pickle('metadata_v2.pkl')
    orthokinesis = pd.read_pickle('orthokinesis_dur_3_slow_-0.7_dis_0.4.pkl')
    conc_orthokinesis = pd.read_pickle('orthokinesis_dur_3_slow_-0.7_dis_0.4_conc_conc_s3.pkl')
    potential_weathervane = pd.read_pickle('weathervane_dur_3_bear_-0.4.pkl')
    potential_curve = pd.read_pickle('curve_dur_3.pkl')
    sharp_turn = pd.read_pickle('sharp_turn_dur_1.5_angle_90.pkl')

    # print(data.columns)
    # exit()

    # before_time = 3
    # lines = condition_filter(just_before_found(found_before, before_time), '12hr')
    # lines = condition_filter(data, '72hr')
    # lines = condition_filter(found_before,'Fed')
    # lines = lines[lines['on_edge'] == 0]
    # lines = lines.reset_index(drop=True)
    # *********************************************************************************************************************
    # potential_sharp_turn = sharp_turn_screen(data, sharp_turn_duration)
    # potential_sharp_turn.to_pickle('sharp_turn_dur_{}_angle_{}.pkl'.format(sharp_turn_duration, angle_threshold))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(potential_sharp_turn)
    # draw_single_trajectory(screen2lines(data, condition_filter(potential_sharp_turn, 'Fed')))
    # exit()
    # *********************************************************************************************************************

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

    # *********************************************************************************************************************

    # g = sns.jointplot(x="speed", y="distance", data=lines, kind="kde", xlim=(0, 200), ylim=(0, 300))
    # # g = sns.jointplot(x="angle", y="delta_distance", data=lines, kind="kde", xlim=(-180, 180), ylim=(-100, 100))
    # # g = sns.jointplot(x="angle", y="distance", data=lines, kind="kde", xlim=(-180, 180), ylim=(0, 300))
    #
    # g.fig.suptitle('{}_{}'.format(lines.loc[0, 'condition'], before_time))

    # ********************************** Orthokinesis Graph Generator **********************************************

    # meta_update(meta,conc_orthokinesis)
    # exit()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta_v1)
    # exit()
    # p_value = np.round(ttest_ind(meta_v1[meta_v1['startvation'] == 'long']['orthokinesis'], meta_v1[meta_v1['startvation'] == 'short']['orthokinesis'])[1],100)
    # print(p_value)

    # sns.set(style="ticks", palette="pastel")
    # # sns.barplot(x="startvation", y="orthokinesis", hue="found", palette=["m", "g"], data=meta_v1, ci=68, capsize=.1)
    # sns.barplot(x="condition", y="orthokinesis", data=meta_v1, ci=68, capsize=.1)
    # # sns.barplot(x="startvation", y="orthokinesis", data=meta_v1, ci=68, capsize=.1)
    # sns.despine(offset=10, trim=True)

    # x1, x2 = 1, 2
    # y, h, col = 1.2 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)


    # ********************************** Sharp Turn Graph Generator **********************************************

    # meta_update(meta, orthokinesis, sharp_turn)
    # exit()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(meta_v2)
    # exit()
    # p_value = np.round(ttest_ind(meta_v1[meta_v1['startvation'] == 'long']['orthokinesis'], meta_v1[meta_v1['startvation'] == 'short']['orthokinesis'])[1],100)
    # print(p_value)

    sns.set(style="ticks", palette="pastel")
    # sns.barplot(x="startvation", y="sharp_turn", hue="found", palette=["m", "g"], data=meta_v2, ci=68, capsize=.1)
    # sns.barplot(x="condition", y="sharp_turn", data=meta_v2, ci=68, capsize=.1)
    sns.barplot(x="startvation", y="sharp_turn", data=meta_v2, ci=68, capsize=.1)
    sns.despine(offset=10, trim=True)

    # x1, x2 = 1, 2
    # y, h, col = 1.2 + 0.05, 0.05, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color=col)


    # *********************************************************************************************************************

    plt.show()
    exit()

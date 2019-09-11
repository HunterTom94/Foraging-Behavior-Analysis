import pandas as pd
import numpy as np
from time import time


def meta_found_filter(df, bool):
    if bool == 1:
        return df.dropna(subset=['find_time'])
    elif bool == 0:
        return df.loc[~df.index.isin(df.dropna(subset=['find_time']).index)]


def condition_filter(df, condition):
    return df[df['condition'] == condition]


def data_index_filter(df, index_ls):
    return df.loc[df['trajectory_index'].isin(index_ls)].sort_values(by=['trajectory_index', 'line_ID'])


def before_found(data, meta):
    found = meta_found_filter(meta, 1)
    for _, row in found.iterrows():
        condition_data = condition_filter(data, row['condition'])
        temp_data = condition_data[condition_data['trajectory_index'] == row['trajectory_index']]
        data = data.drop(temp_data.loc[temp_data['start_time'] > row['find_time']].index)
    data.sort_values(by=['condition', 'trajectory_index', 'line_ID']).reset_index(drop=True).to_pickle(
        'selected_found_before.pkl')


def save_trajectory(data_name, sheet):
    data = pd.read_excel(data_name, sheet_name=sheet,
                         header=None,
                         index_col=False,
                         keep_default_na=True
                         )

    assert data.shape[1] % 3 == 0
    for fly_ind in range(data.shape[1] // 3):
        data_out = data.iloc[3:, fly_ind * 3:(fly_ind + 1) * 3].dropna()
        data_out.to_pickle('trajectories\\{}_{}.pkl'.format(sheet, fly_ind))


def if_found_pkl_gen(data, meta):
    data_copy = data.copy()
    found = meta_found_filter(meta, 1)
    # not_found = meta_found_filter(meta, 0)

    for condition in found['condition'].unique():
        condition_meta = condition_filter(found, condition)
        condition_data = condition_filter(data, condition)
        temp_data = condition_data[
            condition_data['trajectory_index'].isin(condition_meta['trajectory_index'].tolist())]
        data = data.drop(temp_data.index)

    data.sort_values(by=['condition', 'trajectory_index', 'line_ID']).reset_index(drop=True).to_pickle(
        'selected_not_found.pkl')
    data_copy.loc[~data_copy.index.isin(data.index)].sort_values(
        by=['condition', 'trajectory_index', 'line_ID']).reset_index(drop=True).to_pickle('selected_found.pkl')


def just_before_found(found_before, time):
    out_df = pd.DataFrame()
    for condition in found_before['condition'].unique():
        condition_df = condition_filter(found_before, condition)
        for index in condition_df['trajectory_index'].unique():
            index_df = data_index_filter(condition_df, [index])
            out_df = out_df.append(index_df[index_df['start_time'] >= index_df['start_time'].max() - time],
                                   ignore_index=True)

    return out_df

def meta_update(meta, orthokinesis):
    meta['orthokinesis'] = 0
    meta['found'] = 'not_found'
    meta.at[~pd.isnull(meta['find_time']),'found'] = 'found'
    meta['startvation'] = 'no'
    meta.at[meta['condition'].isin(['12hr','24hr','36hr']),'startvation'] = 'short'
    meta.at[meta['condition'].isin(['48hr', '60hr', '72hr']), 'startvation'] = 'long'
    for condition in np.unique(orthokinesis['condition']):
        for trajectory_index in np.unique(orthokinesis['trajectory_index']):
            temp_ortho = orthokinesis[(orthokinesis['condition'] == condition) & (orthokinesis['trajectory_index'] == trajectory_index)]
            meta.at[(meta['condition'] == condition) & (meta['trajectory_index'] == trajectory_index), 'orthokinesis'] = temp_ortho.shape[0]
    meta.to_pickle('metadata_v1.pkl')




# *********************************************************************************************************************
# CHECK METADATA MATCH DATA
# start_time = time()
# data = pd.read_excel('data.xlsx', sheet_name='Fed',
#                          header=None,
#                          index_col=False,
#                          keep_default_na=True
#                          )
# name_list_data = data.iloc[0,:].dropna().reset_index(drop=True)
# print(time()-start_time)
# meta = pd.read_pickle('metadata.pkl')
# temp = meta[meta['conditions'] == 'Fed'].reset_index(drop=True)
# name_list_meta = temp['fly_name']
# count = 0
# print(time()-start_time)
# while count <= np.max([len(name_list_data),len(name_list_meta)]):
#     print(count)
#     if not name_list_data[:count].equals(name_list_meta[:count]):
#         print('wrong at {}'.format(count))
#         exit()
#     count += 1
# print('all ok')
# exit()
# *********************************************************************************************************************

# save_trajectory('data_changed.xlsx','Fed')
# meta = pd.read_pickle('metadata.pkl')
# temp = meta[meta['conditions'] == '60hr'].reset_index(drop=True)
# print(temp.shape)

# exit()

# *********************************************************************************************************************

def extract_meta(data_name):
    data = pd.read_excel(data_name, sheet_name=0,
                         header=None,
                         index_col=False,
                         keep_default_na=True
                         )

    concat_df = pd.DataFrame()
    for condition_ind in range(data.shape[1] // 3):
        temp_df = data.iloc[1:, condition_ind * 3:(condition_ind + 1) * 3].dropna(how='all')
        temp_df.columns = ['fly_name', 'find_time', 'first_50s_speed']
        temp_df['condition'] = data.iloc[0, condition_ind * 3]
        temp_df = temp_df.reset_index(drop=True)
        temp_df['trajectory_index'] = temp_df.index
        concat_df = concat_df.append(temp_df, ignore_index=True)

    concat_df.to_pickle('metadata.pkl')

if __name__ == '__main__':
    data = pd.read_pickle('all_selected_lines.pkl')
    meta = pd.read_pickle('metadata.pkl')
    if_found_pkl_gen(data, meta)
    before_found(data, meta)
    exit()

    extract_meta('ForagingBehaviorMeta_changed.xlsx')

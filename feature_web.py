import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBRegressor
wa_a = pd.concat([wa_train,wa_test_a],axis=0)
wa_name_list = wa.groupby(['wa_name'])['uid'].count().sort_values(ascending=False).reset_index()['wa_name'][0:1000].values
wa_each_name_count=wa[wa.wa_name.map(lambda x: x in wa_name_list)].groupby(['uid','wa_name'])['uid'].count().unstack().add_prefix('wa_each_name_count_').reset_index().fillna(0)

def get_wa_name_feature(wa_data):
    wa_name_unique_cnt = wa_data.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_')
    wa_name_unique_cnt_web = wa_data[wa_data['wa_type'] == 0].groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_web_')
    wa_name_unique_cnt_app = wa_data[wa_data['wa_type'] == 1].groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('wa_name_app_')
    wa_name_unique_cnt = pd.concat([wa_name_unique_cnt, wa_name_unique_cnt_web, wa_name_unique_cnt_app], axis = 1)
    return wa_name_unique_cnt

def get_wa_name_deep_feature(wa_data):
    tmp0 = pd.DataFrame(wa_data[(wa_data['uid'] >= 'u4100') & (wa_data['uid'] < 'u5000')].groupby(['wa_name', 'uid'])['wa_name'].count().groupby(['wa_name']).count())
    tmp1 = pd.DataFrame(wa_data[wa_data['uid'] < 'u4100'].groupby(['wa_name', 'uid'])['wa_name'].count().groupby(['wa_name']).count())
    tmp = pd.concat([tmp0,tmp1],axis = 1)
    tmp.columns = ['bad_times', 'good_times']
    tmp.fillna(0, inplace = True)
    tmp['bad_ratio'] = tmp['bad_times'] / 900
    tmp['good_ratio'] = tmp['good_times'] / 4100
    tmp['diff'] = tmp['bad_ratio'] - tmp['good_ratio']
    tmp[(tmp['bad_times'] >= 35) & (tmp['good_times'] <= 70)].sort_values(by='diff', ascending=False)
    bad_wa_names = list(tmp[(tmp['bad_times'] >= 35) & (tmp['good_times'] <= 70)].index)
    bad_wa_names_cnt = []
    for name in bad_wa_names:
        bad_wa_names_cnt.append(wa_data[wa_data['wa_name'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('bad_wa_names_' + name + '_cnt_'))
    bad_wa_names_cnt = pd.concat(bad_wa_names_cnt, axis = 1)
    bad_wa_names_cnt.fillna(0, inplace=True)
    t = pd.DataFrame(wa_data['wa_name'].value_counts())
    hot_wa_names = list(t[t['wa_name'] > 50000].index)
    hot_wa_names_cnt = []
    for  name in hot_wa_names:
        hot_wa_names_cnt.append(wa_data[wa_data['wa_name'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('hot_wa_names_' + name + '_cnt_'))
    hot_wa_names_cnt = pd.concat(hot_wa_names_cnt, axis = 1)
    hot_wa_names_cnt.fillna(0, inplace=True)
    wa_names_deep = pd.concat([bad_wa_names_cnt, hot_wa_names_cnt], axis = 1)   
    return wa_names_deep

def get_other_feature(wa_data):
    wa_visit_day_cnt = wa_data.groupby(['uid', 'wa_date'])['wa_date'].count().groupby(['uid']).count() #有几天使用了流量
    wa_data['visit_per_dura'] = wa_data['visit_dura'] / wa_data['visit_cnt']
    wa_data['flow_amount'] = wa_data['up_flow'] + wa_data['down_flow']
    wa_visit_per_dura_static = wa_data.groupby(['uid', 'wa_date'])['visit_per_dura'].sum().groupby(['uid']).agg(
    ['sum', 'std', 'max', 'mean', 'median']).add_prefix('wa_visit_per_dura_')
    wa_data['download_speed'] = (wa_data['down_flow'] + 1) / (wa_data['visit_dura'] + 1)
    wa_data['upload_speed'] = (wa_data['up_flow'] + 1) / (wa_data['visit_dura'] + 1)
    wa_data['amount_speed'] = wa_data['download_speed'] + wa_data['upload_speed']
    wa_download_speed_static = wa_data.groupby(['uid'])['download_speed'].agg(['std','max','median','mean','sum']).add_prefix('download_speed_')
    wa_upload_speed_static = wa_data.groupby(['uid'])['upload_speed'].agg(['std','max','median','mean','sum']).add_prefix('upload_speed_')
    wa_amount_speed_static = wa_data.groupby(['uid'])['amount_speed'].agg(['std','max','median','mean','sum']).add_prefix('amount_speed_')
    wa_speed_static = pd.concat([wa_download_speed_static, wa_upload_speed_static, wa_amount_speed_static], axis = 1)
    others = pd.concat([wa_visit_day_cnt, wa_visit_per_dura_static, wa_speed_static], axis = 1)
    return others


def get_wa_feature_matrix():
    wa_data = read_data()
    other_feature = get_other_feature(wa_data)
    wa_name_feature = get_wa_name_feature(wa_data)
    wa_name_deep_feature = get_wa_name_deep_feature(wa_data)
    wa_cnt_feature = get_wa_cnt_feature(wa_data)
    wa_visit_cnt_feature = get_wa_visit_cnt_feature(wa_data)
    wa_visit_dura_feature = get_wa_visit_dura_feature(wa_data)
    up_down_flow_feature = get_up_down_flow_feature(wa_data)
    wa_date_category = get_wa_date_category_feature(wa_data)
    wa_visit_dura_cat = get_visit_dura_cat(wa_data)
    wa = pd.concat([wa_name_feature, wa_name_deep_feature, wa_cnt_feature, wa_visit_cnt_feature, 
    wa_visit_dura_feature, wa_visit_dura_cat, up_down_flow_feature, wa_date_category, other_feature], axis = 1)
    wa.index.name = 'uid'
    wa.reset_index(inplace=True)
    return wa
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
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index().fillna(0)
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index().fillna(0)
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

sms_opp_len_opp_head_unique=sms.groupby(['uid','opp_len'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('sms_opp_len_opp_head_unique_').reset_index().fillna(0)


sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


sms_in_out['sms_in_out_0_rate'] = sms_in_out['sms_in_out_0'] / sms_opp_num['sms_opp_num_count']

sms_in_out['sms_in_out_1_rate'] = sms_in_out['sms_in_out_1'] / sms_opp_num['sms_opp_num_count']

sms['hour'] = sms.start_time.map(lambda x: x[2:4])
sms['day'] = sms.start_time.map(lambda x: x[0:2])

opp_len = [5,7,8,9,10,11,12,13,14]
sms_opp_count = []
for l in opp_len:
    temp = sms[sms.opp_len==l].groupby(['uid','hour'])['uid'].count().unstack().add_prefix('sms_hour_count_opp_len_'+str(l)+'_').reset_index().fillna(0)
    sms_opp_count.append(temp)
    
for l in opp_len:
    temp = sms[sms.opp_len==l].groupby(['uid','day'])['uid'].count().unstack().add_prefix('sms_day_count_opp_len_'+str(l)+'_').reset_index().fillna(0)
    sms_opp_count.append(temp)

sms_each_opp_head_count=sms.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('sms_each_opp_head_count_').reset_index().fillna(0)
sms_opp_num_list = sms.groupby(['opp_num'])['uid'].count().sort_values(ascending=False).reset_index()['opp_num'][0:1000].values
sms_each_opp_num_count=sms[sms.opp_num.map(lambda x: x in sms_opp_num_list)].groupby(['uid','opp_num'])['uid'].count().unstack().add_prefix('sms_each_opp_num_count_').reset_index().fillna(0)


sms_sort = sms.sort_values(by=['uid','start_time'],ascending='True').reset_index()
sms_sort['last_start_time']=sms_sort.groupby(['uid'])['start_time'].apply(lambda i:i.shift(1))
sms_sort['last_start_gap_time'] = sms_sort[['last_start_time','start_time']].apply(lambda x: time_gap(x[0],x[1]),axis=1)
sms_last_start_gap_time=sms_sort.groupby(['uid'])['last_start_gap_time'].agg(['std','max','min','median','mean','sum',np.ptp]).add_prefix('sms_last_start_gap_time_').reset_index()
sms_each_day_count = sms.groupby(['uid','day'])['opp_num'].count().unstack().fillna(0).add_prefix('sms_each_day_count_').reset_index()
sms_each_day_unique_count_opp_head = sms.groupby(['uid','day'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().fillna(0).add_prefix('sms_each_day_unique_count_opp_head_').reset_index()
sms_each_day_unique_count_opp_num = sms.groupby(['uid','day'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().fillna(0).add_prefix('sms_each_day_unique_count_opp_num_').reset_index()
sms_each_day_in_out_0_count = sms[sms.in_out==0].groupby(['uid','day'])['uid'].count().unstack().fillna(0).add_prefix('sms_each_day_in_out_0_count_').reset_index()
sms_each_day_in_out_1_count = sms[sms.in_out==1].groupby(['uid','day'])['uid'].count().unstack().fillna(0).add_prefix('sms_each_day_in_out_1_count_').reset_index()
sms_each_hour_unique_count_opp_head = sms.groupby(['uid','hour'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().fillna(0).add_prefix('sms_each_hour_unique_count_opp_head_').reset_index()
sms_each_hour_unique_count_opp_num = sms.groupby(['uid','hour'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().fillna(0).add_prefix('sms_each_hour_unique_count_opp_num_').reset_index()
sms_in_out_opp_num_unique = sms.groupby(['uid','in_out'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('sms_in_out_opp_num_unique_').reset_index().fillna(0)
sms_in_out_opp_head_unique = sms.groupby(['uid','in_out'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('sms_in_out_opp_head_unique_').reset_index().fillna(0)

def get_msg_opp_num_feature(msg_data):
    msg_opp_num_cnt = pd.DataFrame(msg_data['smg_opp_num'].value_counts())
    hot_msg_num = list(msg_opp_num_cnt[(msg_opp_num_cnt['smg_opp_num'] < 3000) & (msg_opp_num_cnt['smg_opp_num'] > 1000)].index)
    hot_msg_opp_num_cnt = []
    for  name in hot_msg_num:
        hot_msg_opp_num_cnt.append(msg_data[msg_data['smg_opp_num'] == name].groupby(['uid'])['uid'].agg(['count']).add_prefix('hot_msg_opp_num_' + name + '_cnt_'))     
    hot_msg_opp_num_cnt.append(msg_data[msg_data['smg_opp_num'].apply(
        lambda x: x not in hot_msg_num)].groupby(['uid'])['uid'].agg(['count']).add_prefix('other_msg_opp_num_' + '_cnt_'))
    hot_msg_opp_num_cnt = pd.concat(hot_msg_opp_num_cnt, axis = 1)
    hot_msg_opp_num_cnt.fillna(0, inplace=True)
    return hot_msg_opp_num_cnt


def get_msg_short_num_kind(smg_data) :
    short_num_kind = smg_data[smg_data['smg_opp_len']  <= 8].groupby(['uid', 'smg_opp_num'])['smg_opp_num'].count().unstack('smg_opp_num')
    short_num_kind.fillna(0, inplace=True)
    short_num_kind = short_num_kind.apply(lambda x: x.astype(int))
    return short_num_kind

def get_msg_opp_len_div(smg_data):
    smg_opp_cnt = smg_data.groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_')
    smg_opp_less_5_cnt = smg_data[smg_data['smg_opp_len'] <= 5].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_less_5_')
    smg_opp_less_11_cnt = smg_data[(smg_data['smg_opp_len'] < 11) & (smg_data['smg_opp_len'] > 5)].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_less_11_')
    smg_opp_equal_11_cnt = smg_data[smg_data['smg_opp_len'] == 11].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_equal_11_')
    smg_opp_over_11_cnt = smg_data[smg_data['smg_opp_len'] > 11].groupby(['uid'])['smg_opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('smg_opp_over_11_')
    smg_opp_len_div_cnt = pd.concat([smg_opp_cnt, smg_opp_less_5_cnt,smg_opp_less_11_cnt,smg_opp_equal_11_cnt,smg_opp_over_11_cnt], axis = 1)
    smg_opp_len_div_cnt.fillna(0, inplace=True)
    smg_opp_less_5_ratio = pd.concat([smg_opp_less_5_cnt['smg_opp_less_5_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_less_5_cnt['smg_opp_less_5_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_less_11_ratio = pd.concat([smg_opp_less_11_cnt['smg_opp_less_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_less_11_cnt['smg_opp_less_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_equal_11_ratio = pd.concat([smg_opp_equal_11_cnt['smg_opp_equal_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_equal_11_cnt['smg_opp_equal_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_over_11_ratio = pd.concat([smg_opp_over_11_cnt['smg_opp_over_11_unique_count'] / smg_opp_cnt['smg_opp_unique_count'], smg_opp_over_11_cnt['smg_opp_over_11_count'] / smg_opp_cnt['smg_opp_count']], axis = 1)
    smg_opp_len_ratio = pd.concat([smg_opp_less_5_ratio, smg_opp_less_11_ratio, smg_opp_equal_11_ratio, smg_opp_over_11_ratio], axis= 1)
    smg_opp_len_ratio.columns = ['smg_opp_less_5_unique_ratio', 'smg_opp_less_5_cnt_ratio', 'smg_opp_less_11_unique_ratio', 'smg_opp_less_11_cnt_ratio',
    'smg_opp_equal_11_unique_ratio', 'smg_opp_equal_11_cnt_ratio', 'smg_opp_over_11_unique_ratio', 'smg_opp_over_11_cnt_ratio']
    smg_opp_len_div = pd.concat([smg_opp_len_div_cnt ,smg_opp_len_ratio], axis = 1)
    return smg_opp_len_div


def get_msg_feature_matrix():
    msg_data = read_data()

    smg_opp_len_div = get_msg_opp_len_div(msg_data)
    smg_in_and_out = get_msg_in_and_out(msg_data)
    foreign_fre = get_smg_foreign_fre(msg_data)
    msg_operator_div = get_msg_operator_div(msg_data)
    others = get_smg_other_feature(msg_data)
    msg_opp_head_catagory = get_msg_opp_head_catagory(msg_data)
    msg_opp_len_catagory = get_msg_opp_len_catagory(msg_data)
    msg_hour_feature = get_msg_hour_feature(msg_data)
    msg_date_feature = get_msg_date_feature(msg_data)
    msg_opp_num_feature = get_msg_opp_num_feature(msg_data)
    smg = pd.concat([smg_opp_len_div, smg_in_and_out, foreign_fre, msg_operator_div, others, 
        msg_opp_head_catagory, msg_opp_len_catagory, msg_date_feature],axis =1)
    smg['smg_fre'] = smg['smg_in_cnt'] + smg['smg_out_cnt']
    smg['home_fre'] = smg['smg_fre'] - smg['foreign_fre'] 
    smg.fillna(0, inplace=True)
    smg = smg.apply(lambda x: x.astype(int))
    smg.index.name = 'uid'
    smg.reset_index(inplace=True)
    return smg

def local_train_and_test(smg):
    train_in_smg = label_train.merge(smg, how='left', left_on='uid',right_on = 'uid')
    train_in_smg.fillna(0, inplace=True)
    msg_X_train =train_in_smg.drop(['label'],axis=1)
    msg_y_train  = train_in_smg.label
    msg_local_train_X, msg_local_test_X, local_train_Y, local_test_Y = model_selection.train_test_split(msg_X_train, msg_y_train, test_size=0.25, random_state=42)
    local_test_uid = list(msg_local_test_X['uid'])
    msg_local_train_X = msg_local_train_X.drop(['uid'],axis=1)
    msg_local_test_X = msg_local_test_X.drop(['uid'],axis=1)
    msg_model = msg_train_model(msg_local_train_X, local_train_Y)
    msg_prob = msg_model.predict_proba(msg_local_test_X)
    return local_test_Y, msg_prob

def train(smg):
    train_in_smg = label_train.merge(smg, how='left', left_on='uid',right_on = 'uid')
    train_in_smg.fillna(0, inplace=True)
    test_in_smg = label_test.merge(smg, how='left', left_on='uid',right_on = 'uid')
    test_in_smg.fillna(0, inplace=True)
    msg_X_train = train_in_smg.drop(['label', 'uid'],axis=1)
    msg_X_test = test_in_smg.drop(['uid'],axis=1)
    msg_y_train  = train_in_smg.label
    msg_model = msg_train_model(msg_X_train, msg_y_train) 
    msg_prob = msg_model.predict_proba(msg_X_test)
    test_uid = list(label_test['uid'])
    msg_res = list(map(lambda x,y: [x, y], test_uid, list(map(lambda x: x[1], msg_prob))))
    return msg_res
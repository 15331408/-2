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
voice_opp_num = single_groupby(voice,'uid','opp_num',{'unique_count': lambda x: len(pd.unique(x)),'count':'count'},'voice_opp_num_',0)
voice_opp_head = single_groupby(voice,'uid','opp_head',{'unique_count': lambda x: len(pd.unique(x))},'voice_opp_head_',0)

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_opp_len_opp_num_unique=voice.groupby(['uid','opp_len'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('voice_opp_len_opp_num_unique_').reset_index().fillna(0)



voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
voice_call_type_opp_num_unique=voice.groupby(['uid','call_type'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('voice_call_type_opp_num_unique_').reset_index().fillna(0)

voice_in_out_opp_num_unique = voice.groupby(['uid','in_out'])['opp_num'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('voice_in_out_opp_num_unique_').reset_index().fillna(0)

voice_each_opp_head_count=voice.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('voice_each_opp_head_count_').reset_index().fillna(0)

voice_opp_len_opp_head_unique=voice.groupby(['uid','opp_len'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().add_prefix('voice_opp_len_opp_head_unique_').reset_index().fillna(0)

opp_num_list = voice.groupby(['opp_num'])['uid'].count().sort_values(ascending=False).reset_index()['opp_num'][0:1000].values
voice_each_opp_num_count=voice[voice.opp_num.map(lambda x: x in opp_num_list)].groupby(['uid','opp_num'])['uid'].count().unstack().add_prefix('voice_each_opp_num_count_').reset_index().fillna(0)


voice_data['tel_len'] = (((voice_data['tel_end_time'] //1e6 - voice_data['tel_start_time'] //1e6) * 24 + \
(voice_data['tel_end_time'] //1e4 % 100 - voice_data['tel_start_time'] //1e4 % 100)) * 60 + \
(voice_data['tel_end_time'] //1e2 % 100 - voice_data['tel_start_time'] //1e2 % 100)) * 60 + \
(voice_data['tel_end_time'] % 100 - voice_data['tel_start_time'] % 100)
voice_data['tel_len'] = voice_data['tel_len'].astype(int)
voice_data['tel_date'] = voice_data['tel_start_time'].apply(lambda x: int(x // 1e6))
voice_data['tel_hour'] = voice_data['tel_start_time'].apply(lambda x: int(x // 1e4 % 100))
voice['gap_time']=voice[['start_time','end_time']].apply(lambda x: time_gap(x[0],x[1]),axis=1)
voice_gap_time=voice.groupby(['uid'])['gap_time'].agg(['std','max','min','median','mean','sum',np.ptp]).add_prefix('voice_gap_time_').reset_index
voice_sort = (voice.sort_values(by=['start_time','end_time'],ascending=True)).reset_index()
voice_sort['last_end_time']=voice_sort.groupby(['uid'])['end_time'].apply(lambda i:i.shift(1))
voice_sort['last_gap_time'] = voice_sort[['last_end_time','start_time']].apply(lambda x: time_gap(x[0],x[1]),axis=1)
voice_last_gap_time=voice_sort.groupby(['uid'])['last_gap_time'].agg(['std','max','min','median','mean','sum',np.ptp]).add_prefix('voice_last_gap_time_').reset_index()
voice['start_day']=voice.start_time.map(lambda x: x[0:2])
voice['end_day']=voice.end_time.map(lambda x: x[0:2])
voice_start_day_count = voice.groupby(['uid','start_day'])['opp_head'].agg(lambda x: len(pd.unique(x))).unstack().fillna(0).add_prefix('voice_start_day_count_').reset_index()
def get_tel_in_and_out(voice_data):
   tel_in_and_out = voice_data.groupby(['uid', 'tel_in_and_out'])['tel_in_and_out'].count().unstack('tel_in_and_out')
   tel_in_and_out.columns=['tel_in','tel_out']
   tel_in_and_out.fillna(0, inplace=True)
   tel_in_and_out = tel_in_and_out.apply(lambda x: x.astype(int))
   tel_in_and_out['tel_fre'] = tel_in_and_out['tel_in'] + tel_in_and_out['tel_out']
   tel_in_and_out['tel_in_out_ratio'] = (tel_in_and_out['tel_out'] + 1) / (tel_in_and_out['tel_in'] + 1)
    return tel_in_and_out
def get_tel_len_max(voice_data):
    tel_out_max = voice_data[voice_data['tel_in_and_out'] == 0].groupby(['uid'])['tel_len'].max()
    tel_in_max = voice_data[voice_data['tel_in_and_out'] == 1].groupby(['uid'])['tel_len'].max()
    tel_len_max = pd.concat([tel_in_max, tel_out_max], join='outer', axis=1)
    tel_len_max.fillna(0, inplace=True)
    tel_len_max = tel_len_max.apply(lambda x: x.astype(int))
    tel_len_max.columns= ['tel_in_max', 'tel_out_max']
    tel_len_max['in_out_max_diff'] = tel_len_max['tel_in_max'] - tel_len_max['tel_out_max']
    return tel_len_max
def get_tel_len_div(voice_data):
    tel_short_fre = voice_data[voice_data['tel_len'] <= 120].groupby(['uid'])['uid'].count()#少于2min次数
    tel_mid_fre = voice_data[(voice_data['tel_len'] > 120) & (voice_data['tel_len'] <= 600)].groupby(['uid'])['uid'].count() #2-10min 次数
    tel_long_fre = voice_data[voice_data['tel_len'] > 600].groupby(['uid'])['uid'].count() #大于10min次数
    tel_len_div = pd.concat([tel_short_fre,tel_mid_fre,tel_long_fre], axis= 1)
    tel_len_div.columns = ['tel_short_fre','tel_mid_fre','tel_long_fre']
    tel_len_div.fillna(0, inplace=True)
    tel_len_div = tel_len_div.apply(lambda x: x.astype(int))
    return tel_len_div
def get_tel_opp_len_div(voice_data):
    tel_opp_less_5 = voice_data[voice_data['tel_opp_len'] <= 5].groupby(['uid'])['uid'].count()
    tel_opp_less_11 = voice_data[(voice_data['tel_opp_len'] < 11) & (voice_data['tel_opp_len'] > 5)].groupby(['uid'])['uid'].count()
    tel_opp_equal_11 = voice_data[voice_data['tel_opp_len'] == 11].groupby(['uid'])['uid'].count()
    tel_opp_over_11 = voice_data[voice_data['tel_opp_len'] > 11].groupby(['uid'])['uid'].count()
    tel_opp_len_div = pd.concat([tel_opp_less_5,tel_opp_less_11,tel_opp_equal_11,tel_opp_over_11], axis = 1)
    tel_opp_len_div.columns = ['tel_opp_less_5','tel_opp_less_11','tel_opp_equal_11','tel_opp_over_11']
    tel_opp_len_div.fillna(0, inplace=True)
    tel_opp_len_div = tel_opp_len_div.apply(lambda x: x.astype(int))
    return tel_opp_len_div



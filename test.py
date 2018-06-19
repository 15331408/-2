import pandas as pd
import numpy as np
# # SKlearn classification models
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import RidgeClassifier
from scipy import sparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str},encoding='utf-8')
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str},encoding='utf-8')
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str},encoding='utf-8')

voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str},encoding='utf-8')
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str},encoding='utf-8')
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str},encoding='utf-8')

voice_test_a = pd.read_csv('../data/voice_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str},encoding='utf-8')
sms_test_a = pd.read_csv('../data/sms_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str},encoding='utf-8')
wa_test_a = pd.read_csv('../data/wa_test_a.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str},encoding='utf-8')

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})

voice = pd.concat([voice_train,voice_test_a,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test_a,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test_a,wa_test],axis=0)
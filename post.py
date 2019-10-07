import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import gc
import os
import sys
import time
warnings.filterwarnings('ignore')

import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))



from joblib import Parallel, delayed

def white_merge(train, test, key_id):
    data_card = data_card_merged_online(train, test, key_id)
    data_card['type'] = data_card['isFraud_list'].apply(fraud_past)
    data_card['record'] = data_card['isFraud_list'].apply(lambda x: x.count(0) + x.count(1))
    
    white = pd.DataFrame(
        fraud_list(data_card.loc[(data_card['type'] == 'link_white') & (data_card['record'] > 1),
                                 'TransactionID_list']), columns = ['TransactionID'])
    white['white_guess'] = 1
    del data_card;gc.collect()
    return white

def black_merge(train, test, key_id):
    data_card = data_card_merged_online(train, test, key_id)
    data_card['type'] = data_card['isFraud_list'].apply(fraud_past)
    data_card['record'] = data_card['isFraud_list'].apply(lambda x: x.count(0) + x.count(1))

    fraud = pd.DataFrame(
        fraud_list(data_card.loc[data_card['type'] == 'link_black','TransactionID_list']), 
        columns = ['TransactionID'])
    fraud['fraud_guess'] = 1
    del data_card;gc.collect()
    return fraud

def special_black_merge(train, test, key_id):
    data_card = data_card_merged_online(train, test, key_id)
    data_card['type'] = data_card['isFraud_list'].apply(fraud_past)
    data_card['record'] = data_card['isFraud_list'].apply(lambda x: x.count(0) + x.count(1))

    fraud = pd.DataFrame(
        fraud_list(data_card.loc[(data_card['type'] == 'link_black')
                                 & (data_card['record'] > 2)
                                 ,'TransactionID_list']), 
        columns = ['TransactionID'])
    fraud['special_fraud_guess'] = 1
    del data_card;gc.collect()
    return fraud

def grey_merge(train, test, key_id):
    data_card = data_card_merged_online(train, test, key_id)
    data_card['type'] = data_card['isFraud_list'].apply(fraud_past)
    data_card['record'] = data_card['isFraud_list'].apply(lambda x: x.count(0) + x.count(1))
    data_card['ratio'] = data_card['isFraud_list'].apply(lambda x: x.count(1)/(x.count(0) + x.count(1)) 
                                                         if (x.count(0) + x.count(1)) > 0 else len(x))
    grey = pd.DataFrame(
        fraud_list(data_card.loc[(data_card['type'] == 'link_grey') 
                                 & (data_card['record'] > 1)
                                 & (data_card['ratio'] >= 0.9),
                                 'TransactionID_list']), columns = ['TransactionID'])
    grey['grey_guess'] = 1
    del data_card;gc.collect()
    return grey

def online_white_parallel(tr, ts, full_list):
    print("begin.")
    res = Parallel(n_jobs=12, backend = 'multiprocessing') \
            (delayed(white_merge)(tr, ts, col) for col in full_list)
    ans = pd.concat(res, axis = 0)
    ans.reset_index(drop = True, inplace = True)
    return ans

def online_grey_parallel(tr, ts, full_list):
    print("begin.")
    res = Parallel(n_jobs=12, backend = 'multiprocessing') \
            (delayed(grey_merge)(tr, ts, col) for col in full_list)
    ans = pd.concat(res, axis = 0)
    ans.reset_index(drop = True, inplace = True)
    return ans

def online_rule_parallel(tr, ts, full_list):
    print("begin.")
    res = Parallel(n_jobs=12, backend = 'multiprocessing') \
            (delayed(black_merge)(tr, ts, col) for col in full_list)
    ans = pd.concat(res, axis = 0)
    ans.reset_index(drop = True, inplace = True)
    return ans

def online_special_black_parallel(tr, ts, full_list):
    print("begin.")
    res = Parallel(n_jobs=12, backend = 'multiprocessing') \
            (delayed(special_black_merge)(tr, ts, col) for col in full_list)
    ans = pd.concat(res, axis = 0)
    ans.reset_index(drop = True, inplace = True)
    return ans

def data_card_merged_online(train, test, col_id):
    used = ['TransactionID', 'Transaction_day'] + col_id
    tr = train[used + ['isFraud']]
    ts = test[used]
    data = pd.concat([tr, ts]).reset_index(drop = True)
    data.isFraud.fillna(-1, inplace = True)
    data = data.groupby(col_id).agg({k:lambda x: list(x)
    for k in ['TransactionID'] + ['isFraud']})
    data.columns = ["{}_list".format(col) for col in data.columns]
    data.reset_index(inplace = True)
    return data


def fraud_past(x):
    # ['isFraud_list']
    zero = x.count(0)
    one = x.count(1)
    minus = x.count(-1)
    if (zero > 0) & (minus > 0) & (one > 0):
        return 'link_grey'
    elif (zero * minus) > 0:
        return 'link_white'
    elif (one * minus) > 0:
        return 'link_black'
    elif len(x) == 1:
        return 'single'
    elif (zero > 0) & (one > 0):
        return 'grey'
    elif zero > 0:
        return 'white'
    elif one > 0:
        return 'black'
    else:
        return 'outlier'

def fraud_list(x):
    id_list = []
    for i in x:
        id_list += i
    return list(id_list)

def online_rule_set(tr, ts, full_list, local = False):
    full_fraud = pd.DataFrame([])
    if local:
        i = 0
        result = pd.DataFrame([])
    for key_id in full_list:
        with timer("kind:{}".format(key_id)):
            print("begin.")
            data_card = data_card_merged_online(tr, ts, key_id)
            data_card['type'] = data_card['isFraud_list'].apply(fraud_past)
            # data_card['type'] = data_card.apply(fraud_past, axis = 1)
            fraud = pd.DataFrame(fraud_list(data_card.loc[data_card['type'] == 'link_black',
                                      'TransactionID_list']), columns = ['TransactionID'])
            fraud['fraud_guess'] = 1
            if local:
                fraud_num = pd.merge(fraud, ts[['TransactionID','isFraud']], 
                                     how = 'inner', on =['TransactionID'])
                print("num:",fraud_num.isFraud.sum())
                print("ratio:",fraud_num.isFraud.mean()) 
                print("total:",fraud_num.isFraud.count())
                result.loc[i,'num'] = fraud_num.isFraud.sum()
                result.loc[i,'ratio'] = fraud_num.isFraud.mean()
                result.loc[i, 'count'] = fraud_num.isFraud.count()
                result.at[[i], 'var'] = pd.Series([key_id])
                i += 1
            fraud = pd.merge(fraud, ts[['TransactionID']], how = 'inner', on =['TransactionID'])
            print("contains fraud num:",fraud.shape[0])
            full_fraud = pd.concat([full_fraud, fraud]).reset_index(drop = True)
            print("total:", full_fraud.shape[0])
            del data_card;gc.collect()
    full_fraud.drop_duplicates(['TransactionID'], inplace = True)
    full_fraud.reset_index(drop = True)
    print("total without replicate:", full_fraud.shape[0])
    if local:
        return full_fraud, result
    return full_fraud


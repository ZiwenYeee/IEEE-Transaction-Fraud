import lightgbm as lgb
import gc
from sklearn.model_selection import TimeSeriesSplit
import datetime
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import gc
import os
import sys
import time
warnings.filterwarnings('ignore')


# strategy one: traditional Time Series Split
def time_series_lgb(tr_, ts_, features, folds = 5):
    tr = tr_.copy()
    ts = ts_.copy()
    tr.fillna(0, inplace = True)
    ts.fillna(0, inplace = True)
    params = {'num_leaves': 400,
          'min_child_weight': 0.034,
          'feature_fraction': 0.37,
          'bagging_fraction': 0.42,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.39,
          'reg_lambda': 0.65,
          'random_state': 47
         }
    i = 0
    tscv = TimeSeriesSplit(n_splits = folds,)
    score_list = []
    valid_list = []
    last_list = []
    rem_list = []
    time_test = np.zeros((test_transaction.shape[0],))
    time_full_test = np.zeros((test_transaction.shape[0],))
    bst = 0
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features

    for train_idx, valid_idx in tscv.split(train_transaction):
        print('############################################################ fold = {}'.format(i + 1))
        print('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    
        i += 1
        X_train, y_train = tr.loc[train_idx, features],\
                           tr.loc[train_idx, 'isFraud']
        X_train.fillna(0, inplace = True)
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data = True)
        X_valid, y_valid = tr.loc[valid_idx, features],\
                           tr.loc[valid_idx, 'isFraud']
        X_valid.fillna(0, inplace = True)
        dvalid = lgb.Dataset(X_valid, y_valid, reference = dtrain, 
                         free_raw_data = True)
        del X_train, y_train;gc.collect()
        remnant = [i for i in range(np.hstack([train_idx,valid_idx]).max(),train_transaction.shape[0] - 1)]
        model = lgb.train(
            params = params,
            train_set = dtrain,
            valid_sets = [dvalid],
            num_boost_round = 3000,
            early_stopping_rounds = 100,
            verbose_eval = 100,
        )
        predictions = model.predict(X_valid[features])
        score = roc_auc_score(y_true=y_valid, y_score=predictions)
        print("time:", valid_idx, "--score:", score)
        score_list.append(score)
        last_period = [i for i in range(492117, 590540)]
        last_valid = train_transaction.loc[last_period]
        last_valid.fillna(0, inplace = True)
        last_prediction = model.predict(last_valid[features])
        last_score = roc_auc_score(last_valid['isFraud'], last_prediction)
        last_list.append(last_score)
        print("last period validation:", last_score)
        time_test += model.predict(ts[features])/folds
        feature_importances['fold_{}'.format(i + 1)] = model.feature_importance()
        
        
    print("time split k-fold mean +- std:{} and {}".format(np.around(np.mean(score_list),5),\
                                                       np.around(np.std(score_list), 5)  ))

    print("time split last period mean +- std:{} and {}".format(np.around(np.mean(last_list),5),\
                                                       np.around(np.std(last_list), 5)  ))

    return time_test, feature_importances



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, TimeSeriesSplit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic

    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

def focal_binary_object(pred, dtrain):
    gamma_indct = 2.5
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    g1 = sigmoid_pred * (1 - sigmoid_pred)
    g2 = label + ((-1) ** label) * sigmoid_pred
    g3 = sigmoid_pred + label - 1
    g4 = 1 - label - ((-1) ** label) * sigmoid_pred
    g5 = label + ((-1) ** label) * sigmoid_pred
    # combine the gradient
    grad = gamma_indct * g3 * robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
           ((-1) ** label) * robust_pow(g5, (gamma_indct + 1))
    # combine the gradient parts to get hessian components
    hess_1 = robust_pow(g2, gamma_indct) + \
             gamma_indct * ((-1) ** label) * g3 * robust_pow(g2, (gamma_indct - 1))
    hess_2 = ((-1) ** label) * g3 * robust_pow(g2, gamma_indct) / g4
    # get the final 2nd order derivative
    hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
            (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1

    return grad, hess

def holdhout_lightgbm(tr, ts, features, holdout_rate = 0.25):
    params = {'num_leaves': 400,
          'min_child_weight': 0.034,
          'feature_fraction': 0.37,
          'bagging_fraction': 0.42,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.39,
          'reg_lambda': 0.65,
          'random_state': 47
         }
    train_shape = int(tr.shape[0] * (1 - holdout_rate))
    tr_idx = [i for i in range(0, train_shape)]
    ts_idx = [i for i in range(train_shape, tr.shape[0])]                                    
    X_train, y_train = tr.loc[tr_idx, features], tr.loc[tr_idx, 'isFraud']
    X_train.fillna(0, inplace = True)
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data = True)
    X_valid, y_valid = tr.loc[ts_idx, features], tr.loc[ts_idx, 'isFraud']
    X_valid.fillna(0, inplace = True)
    dvalid = lgb.Dataset(X_valid, y_valid, reference = dtrain, 
                         free_raw_data = True)
    
    model = lgb.train(
            params = params,
            train_set = dtrain,
            valid_sets = [dvalid],
            num_boost_round = 3000,
            early_stopping_rounds = 100,
            verbose_eval = 300,
        )
    predictions = model.predict(X_valid[features])
    score = roc_auc_score(y_true=y_valid, y_score=predictions)
    print(score)
    iteration = int(model.best_iteration * 1.212)
    del X_train, y_train, X_valid, y_valid, dtrain, dvalid; gc.collect()
    dtrain = lgb.Dataset(tr[features], tr['isFraud'], free_raw_data = True)
    model = lgb.train(
            params = params,
            train_set = dtrain,
            num_boost_round = iteration
        )    
    test_score = model.predict(ts[features])
    
    return score, test_score

def Stratified_lightgbm(tr, ts, features, n_kfolds = 5):
    print('train.shape = {}, test.shape = {}'.format(tr.shape, ts.shape))
    params = {'num_leaves': 400,
          'min_child_weight': 0.034,
          'feature_fraction': 0.37,
          'bagging_fraction': 0.42,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.39,
          'reg_lambda': 0.65,
          'random_state': 47
         }
    n_train, n_test = tr.shape[0], ts.shape[0]
    oof_train, oof_test= np.zeros((n_train,)), np.zeros((n_test,))
    score_list, model_list = [], []
    skf = StratifiedKFold(n_splits = n_kfolds, 
                          shuffle = True, random_state = 777).split(tr[features], tr['isFraud'])
#     skf = KFold(n_splits=5).split(tr[features], tr['isFraud'])
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('############################################################ fold = {} / {}'.format(i + 1, n_kfolds))
        print('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        X_train, y_train = tr.loc[train_idx, features], tr.loc[train_idx, 'isFraud'] 
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data = True)
        del X_train, y_train; gc.collect();
              
        X_valid, y_valid = tr.loc[valid_idx, features], tr.loc[valid_idx, 'isFraud']
        dvalid = lgb.Dataset(X_valid, y_valid, reference = dtrain, free_raw_data = True)
    
        model = lgb.train(
            params = params,
            train_set = dtrain,
            valid_sets = [dvalid],
            num_boost_round = 6000,
            early_stopping_rounds = 100,
            verbose_eval = 500,
        )
    
        oof_train[valid_idx] = model.predict(X_valid)
        oof_test += model.predict(ts[features]) / n_kfolds
        score_list.append(roc_auc_score(y_valid, oof_train[valid_idx]))
        del X_valid, y_valid, dtrain, dvalid, model; gc.collect();
        
    print('score_list_mean = {:.6f}'.format(np.mean(score_list)))
    print("score full train = {:.6f}".format(roc_auc_score(tr['isFraud'], oof_train)))
    return oof_train, oof_test    

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, TimeSeriesSplit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def kfold_lightgbm(tr, ts, features, params_new = None, n_kfolds = 5):
    # tr = tr_.copy()
    # ts = ts_.copy()
    # tr.fillna(0, inplace = True)
    # ts.fillna(0, inplace = True)
    print('train.shape = {}, test.shape = {}'.format(tr.shape, ts.shape))
    params = {'num_leaves': 450,
          'min_child_weight': 0.03,
          'feature_fraction': 0.37,
          'bagging_fraction': 0.42,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.39,
          'reg_lambda': 0.65,
          'random_state': 47,
          # 'seed':0
         }
    if params_new is not None:
        params.update(params_new)
    n_train, n_test = tr.shape[0], ts.shape[0]
    oof_train, oof_test= np.zeros((n_train,)), np.zeros((n_test, 5))
    score_list, model_list = [], []
#     skf = StratifiedKFold(n_splits = n_kfolds, 
#                           shuffle = True, random_state = 777).split(tr[features], tr['isFraud'])
    skf = KFold(n_splits = n_kfolds, random_state=10).split(tr[features], tr['isFraud'])
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('############################################################ fold = {} / {}'.format(i + 1, n_kfolds))
        print('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        X_train, y_train = tr.loc[train_idx, features], tr.loc[train_idx, 'isFraud'] 
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data = True)
        del X_train, y_train; gc.collect();
              
        X_valid, y_valid = tr.loc[valid_idx, features], tr.loc[valid_idx, 'isFraud']
        dvalid = lgb.Dataset(X_valid, y_valid, reference = dtrain, free_raw_data = True)
    
        model = lgb.train(
            params = params,
            train_set = dtrain,
            valid_sets = [dvalid],
            num_boost_round = 4000,
            early_stopping_rounds = 100,
            verbose_eval = 500,
        )
    
        oof_train[valid_idx] = model.predict(X_valid)
        oof_test[:, i] = model.predict(ts[features]) 
        score_list.append(roc_auc_score(y_valid, oof_train[valid_idx]))
        print("period:", valid_idx,", the score is", roc_auc_score(y_valid, oof_train[valid_idx]))
        del X_valid, y_valid, dtrain, dvalid, model; gc.collect();
        
    print('score_list_mean = {:.6f}'.format(np.mean(score_list)))
    print("score full train = {:.6f}".format(roc_auc_score(tr['isFraud'], oof_train)))
    return oof_train, oof_test, score_list  
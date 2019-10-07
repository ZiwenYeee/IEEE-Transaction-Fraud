#utils
from contextlib import contextmanager
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import gc
import time
warnings.filterwarnings('ignore')



def correlation_reduce(df, threshold = 0.8):
    threshold = threshold
    print('Original Training shape: ', df.shape)
    # Absolute value correlation matrix
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) if column not in ['SK_ID_CURR']]
    print('There are %d columns to remove.' % (len(to_drop)))
    df.drop(to_drop,axis = 1, inplace = True)
    print('Training shape: ', df.shape)
    return df

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, cat_feat, nan_as_category = True):
    original_columns = list(df.columns)
#     categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    data = pd.get_dummies(df, columns = cat_feat, dummy_na= nan_as_category)
    new_columns = [c for c in data.columns if c not in original_columns]
    return data, new_columns 

def woe_encoder(tr, ts, features, nan_as_category = True):
    df = tr[features + ['isFraud']]
    data = pd.concat([tr[features], ts[features]]).reset_index(drop = True)
    if nan_as_category:
        df.fillna("missing", inplace = True)
    woe_features = pd.DataFrame([])
    feat = []
    for col in features:
        temp = df.groupby(col)['isFraud'].apply(lambda x: np.log((x == 1).sum()/(x == 0).sum()) )
        woe_features[col + "_woe"] = data[col].map(temp)
        feat.append(col + "_woe")
    return woe_features, feat
    
#feature
def mail_func(tr, ts):
    # tr_shape = tr.shape[0]
    mail_feature = [col for col in tr.columns if "email" in col]
    data = pd.concat([tr[mail_feature], ts[mail_feature]]).reset_index(drop = True)
    data.fillna("miss.miss", inplace = True)
    data['P_R'] = data['P_emaildomain'] + "_" + data['R_emaildomain']
    mail_count = pd.DataFrame([])
    for col in data.columns:
        temp = data.groupby(col)[col].transform('count')
        mail_count[col + "_count"] = temp
    mail_count.columns = ["feat1_" + col for col in mail_count.columns]
    # tr[mail_count.columns] = mail_count[:tr_shape].reset_index(drop = True)
    # ts[mail_count.columns] = mail_count[tr_shape:].reset_index(drop = True)
    return mail_count

def numeric_func(tr, ts):
    # tr_shape = tr.shape[0]
    numeric_feat = ['TransactionAmt']
    data = pd.concat([tr[numeric_feat], ts[numeric_feat]]).reset_index(drop = True)
    data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(int)
    numeric_count = pd.DataFrame([])
    # ['TransactionAmt_decimal']
    for col in numeric_feat:
        temp = data.groupby(col)[col].transform('count')
        numeric_count[col + "_count"] = temp
    numeric_count.columns = ["feat2_" + col for col in numeric_count.columns]
    # tr[numeric_count.columns] = numeric_count[:tr_shape].reset_index(drop = True)
    # ts[numeric_count.columns] = numeric_count[tr_shape:].reset_index(drop = True)
    return numeric_count

from joblib import Parallel, delayed

def identity_func(tr, ts, tr_id, ts_id):
    def inside_trans(df,col):
        return df.groupby(col)[col].transform('count')/df.shape[0]
    
    tr_shape = tr.shape[0]
    identity = pd.concat([tr_id, ts_id]).reset_index(drop = True)
    transaction = pd.concat([tr[['TransactionID']], ts[['TransactionID']]]).reset_index(drop = True)
    identity = pd.merge(transaction, identity, on = ['TransactionID'], how = 'left')
    category_col = [col for col in identity.columns if identity[col].dtype == 'O']
    numeric_col = [col for col in identity.columns if identity[col].dtype != 'O']
    identity[category_col].fillna("missing", inplace = True)
    identity[numeric_col].fillna(-100, inplace = True)
    identity_count = pd.DataFrame([])
    system_feat = [col for col in identity.columns if col not in ['TransactionID']]
    res = Parallel(n_jobs=-1, require='sharedmem', verbose=0) \
            (delayed(inside_trans)(identity, col) for col in system_feat)
    ans = pd.concat(res, axis = 1)
    ans.columns = ["{}_count".format(col) for col in system_feat]
    identity_count[ans.columns] = ans
    temp = identity.groupby(system_feat)[system_feat[0]].transform('count')/identity.shape[0]
    identity_count["system_feature" + "_count"] = temp
    
    identity_count.columns = ["feat3_" + col for col in identity_count.columns]
    # tr[identity_count.columns] = identity_count[:tr_shape].reset_index(drop = True)
    # ts[identity_count.columns] = identity_count[tr_shape:].reset_index(drop = True)
    return identity_count

def addr_func(tr, ts):
    # tr_shape = tr.shape[0]
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    addr = pd.concat([tr[addr_feature], ts[addr_feature]]).reset_index(drop = True)
    addr.fillna(-1, inplace = True)
    addr['addr3'] = addr['addr1'].astype(str) + "_" + addr['addr2'].astype(str)
    
    addr_count = pd.DataFrame([])
    addr_count['addr1'] = addr['addr1']
    addr_count['addr2'] = addr['addr2']
    for col in addr.columns:
        temp = addr.groupby(col)[col].transform('count')
        addr_count[col + "_count"] = temp
    addr_count['addr1_nunique'] = addr.groupby(['addr1'])['addr2'].transform('nunique')/addr.shape[0]
    addr_count['addr2_nunique'] = addr.groupby(['addr1'])['addr1'].transform('nunique')/addr.shape[0]
    
    addr_count.columns = ["feat4_" + col for col in addr_count.columns]
    # tr[addr_count.columns] = addr_count[:tr_shape].reset_index(drop = True)
    # ts[addr_count.columns] = addr_count[tr_shape:].reset_index(drop = True)
    return addr_count

def product_func(tr, ts):
    # tr_shape = tr.shape[0]
    product_feature = ['ProductCD']
    data = pd.concat([tr[product_feature], ts[product_feature]]).reset_index(drop = True)
    
    product = pd.DataFrame([])
    # product_one_hot, product_cat = one_hot_encoder(data, product_feature)
    # product[product_cat] = product_one_hot[product_cat]
    product['ProductCD_count'] = \
    data[['ProductCD']].groupby(['ProductCD'])['ProductCD'].transform('count')/product.shape[0]
    woe_features, woe_feat = woe_encoder(tr, ts, product_feature)
    product[woe_feat] = woe_features
    product.columns = ["feat5_" + col for col in product.columns]
    # tr[product.columns] = product[:tr_shape].reset_index(drop = True)
    # ts[product.columns] = product[tr_shape:].reset_index(drop = True)
    return product

def match_func(tr, ts):
    tr_shape = tr.shape[0]
    m_feature = [col for col in tr.columns if "M" in col] #category
    data = pd.concat([tr[m_feature], ts[m_feature]]).reset_index(drop = True)
    match = pd.DataFrame([])
    
    # match_one_hot, match_cat = one_hot_encoder(data, m_feature)
    # match[match_cat] = match_one_hot[match_cat]
    woe_match, woe_feat = woe_encoder(tr, ts, m_feature)
    match[woe_feat] = woe_match
    match.columns = ["feat6_" + col for col in match.columns]
    # tr[match.columns] = match[:tr_shape].reset_index(drop = True)
    # ts[match.columns] = match[tr_shape:].reset_index(drop = True)
    return match

def card_func(tr, ts):
    # tr_shape = tr.shape[0]
    card_feature = [col for col in tr.columns if "card" in col] #category
    
    card_count = pd.DataFrame([])
    card = pd.concat([tr[card_feature], ts[card_feature]]).reset_index(drop = True)
    
    card_missing = card.isna().astype(int)
    card_missing.columns = [col + "_missing" for col in card_missing]
#     col_to_ratio = list(card_missing.columns)
#     card_missing['card_missing_sum'] = card_missing.sum(axis = 1)
    card_count[card_missing.columns] = card_missing
    
    card.card2.fillna(-1, inplace = True)
    card.card3.fillna(-1, inplace = True)
    card.card4.fillna("missing", inplace = True)
    card.card5.fillna(-1, inplace = True)
    card.card6.fillna("missing", inplace = True)

    card_count['card1'] = card['card1']
    card_count['card2'] = card['card2']
    card_count['card3'] = card['card3']
    card_count['card5'] = card['card5']
    # card_one_hot, card_cat = one_hot_encoder(card, ['card4', 'card6'])
    # card_count[card_cat] = card_one_hot[card_cat]

    for i in card.columns:
        temp = card.groupby(i)[i].transform('count')
        card_count[i + '_count'] = temp
    temp = card.groupby(card_feature)[card_feature[0]].transform('count')
    card_count["_".join(card_feature) + '_count'] = temp 
    woe_features, woe_feat = woe_encoder(tr, ts, ['card4', 'card6'])
    card_count[woe_feat] = woe_features
    
    card_count.columns = ["feat7_" + col for col in card_count.columns]
    # tr[card_count.columns] = card_count[:tr_shape].reset_index(drop = True)
    # ts[card_count.columns] = card_count[tr_shape:].reset_index(drop = True)
    return card_count

def all_category_encoding(tr, ts):
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category
    feat = card_feature + addr_feature + m_feature + ['TransactionAmt'] + ['ProductCD']
    # tr_shape = tr.shape[0]
    cat_encoding = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    cat_encoding.fillna("missing", inplace = True)
    cat_count = pd.DataFrame([])
    cat_list = [
            # m_feature, 
            card_feature + addr_feature, 
            # card_feature + m_feature, 
            # addr_feature + m_feature, 
        
            card_feature + addr_feature + m_feature 
            ]
    for col in cat_list:
        cat_count['_'.join(col) + '_count'] = \
        cat_encoding.groupby(col)[col[0]].transform('count')
        cat_count['_'.join(col) + '_amt_unique'] = \
        cat_encoding.groupby(col)['TransactionAmt'].transform('nunique')
    #maybe overfitting.    
    # cat_count['card_addrM_ratio'] = \
    # cat_count['card1_card2_card3_card4_card5_card6_addr1_addr2_M1_M2_M3_M4_M5_M6_M7_M8_M9_count']/\
    # cat_count['card1_card2_card3_card4_card5_card6_M1_M2_M3_M4_M5_M6_M7_M8_M9_count']
    # cat_count['card_addr_ratio'] = \
    # cat_count['card1_card2_card3_card4_card5_card6_addr1_addr2_M1_M2_M3_M4_M5_M6_M7_M8_M9_count']/\
    # cat_count['card1_card2_card3_card4_card5_card6_addr1_addr2_count']

    cat_count.columns = ["feat8_" + col for col in cat_count.columns]

    # tr[cat_count.columns] = cat_count[:tr_shape].reset_index(drop = True)
    # ts[cat_count.columns] = cat_count[tr_shape:].reset_index(drop = True)
    return cat_count

    
def Transaction_amt_encoding(tr, ts):
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category
    feat = card_feature + addr_feature + m_feature + ['TransactionAmt'] + ['ProductCD']
    # tr_shape = tr.shape[0]
    cat_encoding = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    cat_encoding.fillna("missing", inplace = True)
    cat_count = pd.DataFrame([])
    # cat_count['category_amt_ratio'] = \
    # cat_encoding['TransactionAmt']/cat_encoding.groupby(card_feature + addr_feature + 
    #                                     m_feature)['TransactionAmt'].transform('max')
    cat_count['card_amt_ratio'] = \
    cat_encoding['TransactionAmt']/cat_encoding.groupby(card_feature)['TransactionAmt'].transform('max')

    #maybe overfitting.
    cat_count['product_amt_ratio'] = \
    cat_encoding['TransactionAmt']/cat_encoding.groupby(['ProductCD'])['TransactionAmt'].transform('max')
    
    cat_count['card_addr_amt_ratio'] = \
    cat_encoding['TransactionAmt']/cat_encoding.groupby(card_feature + addr_feature)['TransactionAmt'].transform('max')
    #cumcount *** easy to overfit
    for i in ['min', 'max', 'sum', 'std', 'mean']:
        temp = cat_encoding.groupby(card_feature)['TransactionAmt'].transform(i)
        cat_count[i + '_card_feature_trans'] = temp
        temp = cat_encoding.groupby(addr_feature)['TransactionAmt'].transform(i)
        cat_count[i + '_addr_feature_trans'] = temp
        
        #risk of overfitting.
        temp = cat_encoding.groupby(card_feature + addr_feature)['TransactionAmt'].transform(i)
        cat_count[i + '_card_addr_feature_trans'] = temp
        
        
        # temp = cat_encoding.groupby(card_feature + addr_feature + m_feature)['TransactionAmt'].transform(i)
        # cat_count[i + '_card_addr_m_feature_trans'] = temp        
        #maybe overfitting.
#         temp = cat_encoding.groupby(['ProductCD'])['TransactionAmt'].transform(i)
#         cat_count[i + '_ProductCD_trans'] = temp        
#         cat_count[i + '_card_feature_trans_div'] =  cat_encoding['TransactionAmt']/temp

    cat_count.columns = ["feat9_" + col for col in cat_count.columns]
    
    # tr[cat_count.columns] = cat_count[:tr_shape].reset_index(drop = True)
    # ts[cat_count.columns] = cat_count[tr_shape:].reset_index(drop = True)
    return cat_count

def C_feature(tr, ts):
    c_feature = [col for col in tr.columns if "C" in col] #numeric
    c_feature.remove('ProductCD')
    feat = c_feature + ['TransactionAmt']
    c_feat = [col for col in c_feature if col not in ['C13']]
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category
    
    tr_shape = tr.shape[0]
    C_count = pd.DataFrame([])
    feat = c_feature+ ['TransactionAmt'] + card_feature + addr_feature + m_feature
    C = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    C_count['C_sum'] = C[c_feature].sum(axis = 1)
    for i in c_feature:
        temp = C[i]/C_count['C_sum']
        C_count[i + '_ratio'] = temp
        
    # cat_list = [card_feature + addr_feature + c_feat,
    #             card_feature + addr_feature + m_feature + c_feat]
    # for col in cat_list:
    #     C_count['_'.join(col) + '_count'] = \
    #     C.groupby(col)[col[0]].transform('count')    
    
    C_count.columns = ["feat10_" + col for col in C_count.columns]

#     C_count['_'.join(c_feature) + "_count"] = C.groupby(c_feature)[c_feature[0]].transform('count')/C.shape[0]
    # tr[C_count.columns] = C_count[:tr_shape].reset_index(drop = True)
    # ts[C_count.columns] = C_count[tr_shape:].reset_index(drop = True)
    return C_count


from sklearn.decomposition import TruncatedSVD

def df_to_svd(df, name, n = 5):
    svd = TruncatedSVD(n_components = n, n_iter = 50, random_state = 777)

    df_svd = pd.DataFrame(svd.fit_transform(df.values))
    df_svd.columns = ['svd_{}_{}'.format(name, i) for i in range(n)]
    return df_svd

def pca_missing(tr, ts):
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category
    tr_shape = tr.shape[0]
    feat = addr_feature + m_feature + dist_feature
    pca_pre = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    pca_pre = pca_pre.isna().astype(int)
    pca_pre.columns = [col + '_missing_flag' for col in pca_pre.columns]
    svd = pca_pre
    del pca_pre;gc.collect()
#     svd = df_to_svd(pca_pre, 'missing', 30)
#     del pca_pre;gc.collect()
    # tr[svd.columns] = svd[:tr_shape].reset_index(drop = True)
    # ts[svd.columns] = svd[tr_shape:].reset_index(drop = True)
    return svd
    
def date_feature(tr, ts):
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category
    feat = card_feature + addr_feature + m_feature + ['TransactionDT'] + ['ProductCD'] + ['TransactionAmt']
    data = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    data['Transaction_day'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1))
    data['Transaction_day_of_week'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 7)
    data['Transaction_hour_of_day'] = np.floor(data['TransactionDT'] / 3600) % 24
    
    time = pd.DataFrame([])
    time['day_of_week_count'] = \
    data.groupby(['Transaction_day'])['Transaction_day'].transform('count')   
    #overfitting.
    time['day_of_week_hour_count'] = \
    data.groupby(['Transaction_day', 'Transaction_hour_of_day'])['Transaction_day'].transform('count')
    
    # time['day_of_week_ratio'] = time['day_of_week_hour_count']/time['day_of_week_count']
    # time['day_uid_count'] = \
    # data.groupby(card_feature + addr_feature + ['Transaction_day'])['Transaction_day'].transform('count')
    
    
    for col in ['Transaction_day_of_week', 'Transaction_hour_of_day', 'Transaction_day']:
        time[col + "_unique"] = \
        data.groupby(card_feature + addr_feature)[col].transform('nunique')
        
    time.columns = ["feat11_" + col for col in time.columns]
    return time

def D_feature(tr, ts):
    d_feature = [col for col in tr.columns if "D" in col] #numeric
    d_feature.remove('TransactionID')
    # d_feature.remove('TransactionDT')
    d_feature.remove('ProductCD')
    card_feature = [col for col in tr.columns if "card" in col] #category
    addr_feature = [col for col in tr.columns if "addr" in col] #category
    m_feature = [col for col in tr.columns if "M" in col] #category

    feat = d_feature + card_feature + addr_feature + m_feature
    data = pd.concat([tr[feat], ts[feat]]).reset_index(drop = True)
    data['Transaction_day'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1))
    D = pd.DataFrame([])
    for col in d_feature:
        D[col + '_new'] = data[col] - data['Transaction_day']
        data[col + "_new"] = data[col] - data['Transaction_day']
        if col in ['D1', 'D10', 'D15']:
            D[col + '_delta'] = data[col] - data['Transaction_day']
            D.loc[D[col + '_delta'] >= 0, col + '_delta'] = None
        # D[col + '_flag'] = np.where( data[col] > data['Transaction_day'] - 480, 1, 0)
    # for col in d_drop:
    #     D[col + '_new_count'] = \
    #     data.groupby(card_feature + [col + '_new'])[col + '_new'].transform('count')

    D.columns = ["feat12_" + col for col in D.columns]
    # D[data.columns] = 1 - data.isna().astype(int)
    # D.columns = ["existed_" + col for col in D.columns]
    # D['D_missing_sum'] = data.isna().astype(int).sum(axis = 1)
    # for i in ['card1', 'card4', 'addr1']:
    #     D['D15_to_mean_{}'.format(i)] = data['D15'] / data.groupby([i])['D15'].transform('mean')
    # tr[D.columns] = D[:tr_shape].reset_index(drop = True)
    # ts[D.columns] = D[tr_shape:].reset_index(drop = True)
    return D



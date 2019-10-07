import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import gc
import os
import sys
import time
warnings.filterwarnings('ignore')


def trend_plot(tr, ts, plot_list, fraud_type = False):
    for c in plot_list:
        try:
            if fraud_type:
                tr[tr['isFraud'] == 0].set_index('TransactionDT')[c].plot(style='.', title='Hist ' + c, 
                                                                                       figsize=(15, 3), color = 'blue')
                tr[tr['isFraud'] == 1].set_index('TransactionDT')[c].plot(style='.', title='Hist ' + c, 
                                                                                       figsize=(15, 3), color = 'orange')
                ts.set_index('TransactionDT')[c].plot(style='.', color = 'green',
                                                       title=c + ' values over time (blue=no-fraud, orange=fraud, green=test)', 
                                                       figsize=(15, 3))
                plt.show()
            else:
                tr.set_index('TransactionDT')[c].plot(style='.', title=c, figsize=(15, 3), alpha=0.01)
                ts.set_index('TransactionDT')[c].plot(style='.', title=c, figsize=(15, 3), alpha=0.01)
                plt.show()
        except TypeError:
            pass
        except KeyError:
            pass
        
        

# train_transaction['Transaction_day'] = np.floor((train_transaction['TransactionDT'] / (3600 * 24) - 1))
# test_transaction['Transaction_day'] = np.floor((test_transaction['TransactionDT'] / (3600 * 24) - 1))
# train_transaction['linear'] = train_transaction['Transaction_day'] + 480
# test_transaction['linear'] = test_transaction['Transaction_day'] + 480
# # train_transaction['D10_new'] = train_transaction['D10'] - test_transaction['linear']
# # test_transaction['D10_new'] = test_transaction['D10'] - test_transaction['linear']
# c = 'D15'
# train_transaction[train_transaction['isFraud'] == 0].set_index('TransactionDT')[c].plot(style='.', title='Hist ' + c, 
#                                                         figsize=(15, 3), color = 'blue')
# train_transaction[train_transaction['isFraud'] == 1].set_index('TransactionDT')[c].plot(style='.', title='Hist ' + c, 
#                                                           figsize=(15, 3), color = 'orange')
# train_transaction.set_index('TransactionDT')['linear'].plot(style='.', title='Hist ', 
#                                                               figsize=(15, 3), alpha=1, color = 'red')
# test_transaction.set_index('TransactionDT')[c].plot(style='.', color = 'green',
#                                     title=c + ' values over time (blue=no-fraud, orange=fraud, green=test)', 
#                                       figsize=(15, 3))
# test_transaction.set_index('TransactionDT')['linear'].plot(style='.', color = 'red',
#                                     title=c + ' values over time (blue=no-fraud, orange=fraud, green=test)', 
#                                       figsize=(15, 3))
# plt.show()
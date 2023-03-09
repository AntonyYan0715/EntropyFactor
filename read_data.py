import pandas as pd
import numpy as np
import math
from scipy.io import loadmat
import os.path
import warnings
from tqdm import trange
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

stock_code = loadmat('C:/Users/11765/Desktop/SoochowSecurities/entropy/data/AllStockCode.mat')['AllStockCode']
stock_code = [x[0][:6] for x in stock_code[0]]
trading_date = loadmat('C:/Users/11765/Desktop/SoochowSecurities/entropy/data/TradingDate_Daily.mat')['TradingDate_Daily'].reshape(4272)
daily_share = loadmat('C:/Users/11765/Desktop/SoochowSecurities/entropy/data/AllStock_DailyAShareNum.mat')['AllStock_DailyAShareNum']
daily_share = pd.DataFrame(daily_share, index=trading_date, columns=stock_code)

path = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/股票分钟数据20210730/'
def get_file(path):
    files = os.listdir(path)
    files.sort()
    list = []
    
    for file in files:
        if not os.path.isdir(path + file):  
            f_name = str(file)
            filename = path + f_name
            list.append(filename)
    return (list)

file_list = get_file(path)
stock_list = []
for i in range(len(file_list)):
    stock_list.append(file_list[i].split('/')[-1].split('.')[0])
code_list = [x.split('_')[-1] for x in stock_list]

# ==========================================
# Read the first file, get all trading dates
mat_list = pd.DataFrame((loadmat(file_list[0])[stock_list[0]]),columns = ['date','start','max','min','end','total_quantity','total_amount'])
end_date = 20210730
date_list = list(set(list(mat_list['date'])))
date_list.sort()
date_list = date_list[: date_list.index(end_date)+1]

# ==================================
# Use for-cycle to calculate factors
turnover = pd.DataFrame(index=stock_code, columns=date_list)

for i in trange(len(file_list)):
    # Read a file
    mat_list = pd.DataFrame((loadmat(file_list[i])[stock_list[i]]),columns = ['date','start','max','min','end','total_quantity','total_amount'])
    stock = code_list[i]
    if stock in stock_code:
        
        # Take daily mat (240 data)
        for date in date_list:
            current_mat = mat_list[mat_list['date'] == date]
            current_mat = current_mat[current_mat['total_quantity'] != 0].reset_index(drop = True)
            share = daily_share.loc[date, stock]
            current_mat['turnover_rate'] = current_mat['total_quantity'] / share
            turnover.loc[stock, date] = current_mat.turnover_rate.std()

# ==============
# Get date range
last_dates = []
first_dates = []
date_range = {}   # At last dates of each month, look back 20 days

for i in range(len(date_list) - 1):
    if str(date_list[i])[:6] != str(date_list[i+1])[:6]:
        last_dates.append(date_list[i])
        first_dates.append(date_list[i+1])
        if i > 20:
            date_range[date_list[i]] = []
            for j in range(i-19, i+1):
                date_range[date_list[i]].append(date_list[j])

# =================
# Calculate entropy
entropy = pd.DataFrame(index=stock_code, columns=date_list)

for i in trange(len(file_list)):
    mat_list = pd.DataFrame((loadmat(file_list[i])[stock_list[i]]),columns = ['date','start','max','min','end','total_quantity','total_amount'])
    stock = code_list[i]
    if stock in stock_code:
        
        for date in last_dates[1:]:
            twenty_dates = date_range[date]
            all_mat = pd.DataFrame(columns = ['date','start','max','min','end','total_quantity','total_amount', 'turnover_rate'])
            
            for day in twenty_dates:
                current_mat = mat_list[mat_list['date'] == day]
                current_mat = current_mat[current_mat['total_quantity'] != 0].reset_index(drop = True)
                share = daily_share.loc[day, stock]
                current_mat['turnover_rate'] = current_mat['total_quantity'] / share
                all_mat = pd.concat([all_mat, current_mat])
            
            all_mat = all_mat.reset_index(drop = True)
            all_mat['turnover_rate'] = all_mat['turnover_rate'].replace(np.inf, np.nan)
            Max = all_mat['turnover_rate'].max()
            Min = all_mat['turnover_rate'].min()
            
            if np.isnan(Max)==False and np.isnan(Min)==False:
                for day in twenty_dates:
                    turnover_data = all_mat[all_mat['date'] == day]['turnover_rate']
                    hist = np.histogram(turnover_data, bins=40, range=(Min, Max))[0] / len(turnover_data)
                    H = -np.nansum(hist * np.log2(hist)) / np.log2(40)
                    entropy.loc[stock, day] = H

entropy.to_hdf('C:/Users/11765/Desktop/SoochowSecurities/entropy/result/entropy40.h5', key='df')



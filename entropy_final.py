import numpy as np
import pandas as pd
from scipy.io import loadmat
import os.path
import warnings
from tqdm import trange
warnings.filterwarnings("ignore")

dir_data = 'C:/Users/Administrator/Desktop/data/'
daily_share = pd.read_hdf(dir_data + 'DailyAShareNum.h5', key='df')
stock_code = list(daily_share.columns)
trading_date = list(daily_share.index)

path = dir_data + '股票分钟数据20210730/'
def get_file(path):
    files = os.listdir(path)
    files.sort()
    list1 = []
    
    for file in files:
        if not os.path.isdir(path + file):  
            f_name = str(file)
            filename = path + f_name
            list1.append(filename)
    return list1

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
def calc_entropy(split_num, min_data_num):
    entropy = pd.DataFrame(index=stock_code, columns=last_dates[1:])
    
    for i in trange(1, len(last_dates)):
        date = last_dates[i]
        twenty_dates = date_range[date]
        turnover_mat = pd.DataFrame(columns=stock_code, index=range(4800))
        
        for j in range(len(file_list)):
            stock = stock_list[j][-6:]
            
            if stock in stock_code:
                mat_list = pd.DataFrame((loadmat(file_list[j])[stock_list[j]]),columns = ['date','start','max','min','end','total_quantity','total_amount'])
                turnover_list = []
                
                for day in twenty_dates:
                    current_mat = mat_list[mat_list['date'] == day]
                    current_mat = current_mat[current_mat['total_quantity'] != 0].reset_index(drop = True)
                    share = daily_share.loc[day, stock]
                    current_mat['turnover_rate'] = current_mat['total_quantity'] / share
                    turnover_list = turnover_list + list(current_mat['turnover_rate'].replace(np.inf, np.nan))
                
                if len(turnover_list) <= 4800 and len(turnover_list) >= (min_data_num * 240):
                    turnover_mat[stock][range(len(turnover_list))] = turnover_list
                    
        Max = turnover_mat.max().max()
        Min = turnover_mat.min().min()
        
        for stock in stock_code:
            num = turnover_mat[stock].count()
            
            if num > 0:
                hist = np.histogram(turnover_mat[stock], bins=split_num, range=(Min, Max))[0] / num
                H = -np.nansum(hist * np.log2(hist)) / np.log2(split_num)
                entropy.loc[stock, date] = H
    
    entropy.to_hdf(dir_data + 'result/entropy-' + str(split_num) + '.h5', key='df')

# calc_entropy(1000, 18)
# calc_entropy(1500, 18)
# calc_entropy(2000, 18)
# calc_entropy(2500, 18)

# =======================
# Calculate 4800 data std
Std = pd.DataFrame(index=stock_code, columns=last_dates[1:])

for i in trange(1, len(last_dates)):
    date = last_dates[i]
    twenty_dates = date_range[date]
    
    for j in range(len(file_list)):
        stock = stock_list[j][-6:]
        
        if stock in stock_code:
            mat_list = pd.DataFrame((loadmat(file_list[j])[stock_list[j]]),columns = ['date','start','max','min','end','total_quantity','total_amount'])
            turnover_list = []
            
            for day in twenty_dates:
                current_mat = mat_list[mat_list['date'] == day]
                current_mat = current_mat[current_mat['total_quantity'] != 0].reset_index(drop = True)
                share = daily_share.loc[day, stock]
                current_mat['turnover_rate'] = current_mat['total_quantity'] / share
                turnover_list = turnover_list + list(current_mat['turnover_rate'].replace(np.inf, np.nan))
            
            if len(turnover_list) <= 4800 and len(turnover_list) >= 4320:
                Std.loc[stock, date] = np.std(turnover_list)

Std.to_hdf(dir_data + '4800std.h5', key='df')

        

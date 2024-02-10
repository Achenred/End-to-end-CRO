import yfinance as yf
from functools import reduce
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import sys
import random
import os

seed = 1000 


df_open=pd.DataFrame()
df_volume=pd.DataFrame()
cnt=0
tickers_to_retry = []

ticker_list=['CSCO', 'BA', 'MDT', 'HSBC', 'MO',
        'NVS', 'BCH', 'CHTR', 'C', 'T', 'BAC', 'BP',
        'PEP', 'IEP', 'UL', 'D', 'MRK', 'TSM', 'CODI', 'ORCL',
        'PG', 'CAT', 'MCD', 'AMZN', 'INTC', 'MMM', 'KO', 'NEE', 'UPS', 'MSFT',
        'EXC', 'HD', 'SO', 'XOM', 'CVX', 'CMCSA', 'PCG', 'GOOG'
        , 'NGG', 'BHP',  'GD', 'PM', 'DIS', 'GE', 
        'BSAC', 'JPM', 'DHR',  'SRE', 'GOOG', 'PFE', 'DUK',
        'VZ', 'AMGN', 'SNY', 'UNH', 'MA', 'HON', 'SLB', 'AAPL', 'WMT',
        'LMT', 'AEP', 'JNJ', 'REX', 'PPL'] #, 'BRK-A', 'TM', 'V',


# stocks=pd.read_csv(r'path/data/stocks.csv')

ticker_list_small=  ticker_list #list(stocks.iloc[seed,:])
ticker_list_small.sort()

start="2012-01-01"
end= "2021-12-31"

# Downloading data
data = yf.download(ticker_list_small, start = start, end = end).reset_index()


data_subset = data.loc[:,('Adj Close', slice(None))]
new_cols = data_subset.columns
tickers = [i[1] for i in new_cols]

data_subset.columns = tickers

data_subset = data_subset[tickers].pct_change()  #.dropna()

returns = data_subset
data_subset = - data_subset
data_subset['DATE'] = data.loc[:,('Date', '')]
data_subset.dropna(inplace = True)


returns['DATE'] = data.loc[:,('Date', '')]
returns.dropna(inplace = True)

data_vol = data.loc[:,('Volume', slice(None))]
data_vol.columns = tickers
data_vol['DATE'] = data.loc[:,('Date', '')]
data_vol.dropna(inplace = True)
data_vol = data_vol.add_suffix('_SI')
data_vol = data_vol.rename({'DATE_SI': 'DATE'}, axis=1)
data_vol.dropna(inplace = True)


data_volume=data_vol[data_vol.DATE.isin(data_subset.DATE)]


market_features = ['^DJI', '^VIX', '^GSPC','CL=F', 'GC=F', '^TNX' ]
market_col_names = ['DJI_SI', 'VIX_SI', 'GSPC_SI','CLF_SI', 'GCF_SI', 'TNX_SI' ]
data_m = yf.download(market_features, start = start, end = end).reset_index()
data_market = data_m.loc[:,('Open', slice(None))]


new_market_cols =  [i[1] for i in data_market.columns]

data_market.columns = new_market_cols
data_market = data_market[market_features]
data_market.columns = market_col_names

data_market['DATE'] = data_m.loc[:,('Date', '')]
data_market.dropna(inplace = True)

data_market=data_market[data_market.DATE.isin(data_subset.DATE)]


data_subset.DATE=data_subset.DATE.astype('datetime64[ns]')
data_volume.DATE  =data_volume.DATE.astype('datetime64[ns]')
data_market.DATE=data_market.DATE.astype('datetime64[ns]')

col_order = tickers


volume = True
if volume:
    data_side=pd.merge(data_volume,data_market,on='DATE', how = 'left')
else:
    data_side = data_market
col_order = col_order + list(data_side.columns)
col_order.remove('DATE')

data_final=pd.merge(data_subset,data_side,on='DATE', how = 'left')
data_final.fillna(data_final.mean(), inplace=True)

data_final[col_order] = data_final[col_order].apply(pd.to_numeric, errors='coerce')
col_order = ['DATE'] + col_order
data_final = data_final.reindex(columns=col_order)



data_subset.to_csv(r'data/expected_cost.csv',index=False)
returns.to_csv(r'data/expected_return.csv',index=False)
data_side.to_csv(r'data/side_info.csv',index=False)
data_final.to_csv(r'data/final_port.csv',index=False)



# Set random seed
random_seed = 42

# Iterate over years from 2012 to 2017
for year in range(2012, 2018):
    # Set random seed for each year
    np.random.seed(random_seed + year - 2012)
    
    # Create directory for the current year
    year_directory = f"data/{year}_samples"
    os.makedirs(year_directory, exist_ok=True)
    
    # Generate and save 60 samples in groups of 10
    for group in range(0, 60, 6):
        returns_list = []
        side_list = []
        for _ in range(10):
            sample = np.random.choice(tickers, size=15, replace=False)
            side_sample = [i+"_SI" for i in sample] + market_col_names
            returns_list.append(sample)
            side_list.append(side_sample)
        
   
        # Save the sample group as a text file
        filename = f"{year}_returns_{group // 6}.txt"
        with open(os.path.join(year_directory, filename), 'w') as f:
            for sample in returns_list:
                f.write(','.join(map(str, sample)) + '\n')
                
        # Save the sample group as a text file
        filename = f"{year}_data_side_{group // 6}.txt"
        with open(os.path.join(year_directory, filename), 'w') as f:
            for sample in side_list:
                f.write(','.join(map(str, sample)) + '\n')
                
                
                
                
#%%
import requests
from datetime import datetime, timedelta
import pandas as pd
#%%
def get_historical_data(coin_id, from_timestamp, to_timestamp):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    headers = {
        "x-cg-pro-api-key": "CG-Ryp3NHHj6bx1zo84RvG1vG9x"}
    params = {
        "vs_currency": "usd", 
        "from": date_to_timestamp(from_timestamp), 
        "to": date_to_timestamp(to_timestamp)
    }
    response = requests.get(url, headers=headers, params = params)
    data = response.json()
    return data
def historical_data_to_df(data_dict):
    prices_df = pd.DataFrame(data_dict['prices'], columns= ['time', 'prices'])
    market_cap_df = pd.DataFrame(data_dict['market_caps'], columns = ['time', 'market_cap'])
    volumes_df = pd.DataFrame(data_dict['total_volumes'], columns = ['time', 'volumes'])
    
    df_list = [prices_df, market_cap_df, volumes_df]
    
    for df in df_list:
        df['time'] = pd.to_datetime(df['time'], unit = 'ms')
        
    for df in df_list[1:]:
        df_list[0] = pd.merge(df_list[0], df, on = 'time')
    
    return df_list[0]
def date_to_timestamp(date_string):
    dt = datetime.strptime(date_string, '%Y-%m-%d')
    return dt.timestamp()
#%%
#list
market_rank_list = ["bitcoin"]
#%%
#hourly
def get_hourly_data():
    def get_hourly_data_by_coin(coin_id):
        start_date = datetime(2011, 12, 15)
        end_date = datetime(2023, 5, 30)
        
        days_per_call = 90
        
        current_date = start_date
        data = []
        while current_date <= end_date:
            call_end_date = min(current_date + timedelta(days=days_per_call), end_date)
            try:
                data_dict = get_historical_data(coin_id, current_date.strftime('%Y-%m-%d'), call_end_date.strftime('%Y-%m-%d'))
                df = historical_data_to_df(data_dict)
                data.append(df)
                current_date = call_end_date + timedelta(days=1)
            except:
                continue 
        data_df = pd.concat(data)
        return data_df
    for coin in market_rank_list:
        data_hourly = get_hourly_data_by_coin(coin)
    return data_hourly
#%%
data_hourly = get_hourly_data()
data_hourly = data_hourly[data_hourly['time'] >= "2018-06-06"]
data_hourly.to_csv("btc_hourly",index = False)
#%%
data_hourly.describe()
# %%

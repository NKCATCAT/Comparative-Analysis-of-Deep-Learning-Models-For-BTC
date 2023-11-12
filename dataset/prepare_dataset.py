#%%
import pandas as pd
import pandas_ta as ta
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
#%%
df = pd.read_csv('btc_hourly.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
ohlc = df['prices'].resample('4H').ohlc()
volume = df['volumes'].resample('4H').sum()
ohlc['volumes'] = volume
ohlc.index = ohlc.index + pd.DateOffset(hours=4)
#%%
#get location features
def loc_feature(data):
    loc = []
    for i in range(1,len(data)):
        close1 = data["close"][i] # today
        high1 = data['high'][i]
        low1 = data['low'][i]
        volume1 = data['volumes'][i]

        close0 = data['close'][i-1] # yesterday
        high0 = data['high'][i-1]
        low0 = data['low'][i-1]
        volume0 = data['volumes'][i-1]

        if close1 > close0: 
            if volume1 >  volume0:
                if high1 > high0:
                    if low1 > low0:
                        loc.append(1)
                    else:
                        loc.append(2)
                else:
                    if low1 > low0:
                        loc.append(3)
                    else:
                        loc.append(4)
            else:
                if high1 > high0:
                    if low1 > low0:
                        loc.append(5)
                    else:
                        loc.append(6)
                else:
                    if low1 > low0:
                        loc.append(7)
                    else:
                        loc.append(8)                    
        else: 
            if volume1 >  volume0:
                if high1 > high0:
                    if low1 > low0:
                        loc.append(9)
                    else:
                        loc.append(10)
                else:
                    if low1 > low0:
                        loc.append(11)
                    else:
                        loc.append(12)
                
            else:
                if high1 > high0:
                    if low1 > low0:
                        loc.append(13)
                    else:
                        loc.append(14)
                else:
                    if low1 > low0:
                        loc.append(15)
                    else:
                        loc.append(16)
    loc = pd.DataFrame(loc, index = data.index[1:])
    data["Location"] = loc
    return data

#get rest of the features and output labels
def get_features(dataset):
    dataset['returns'] = dataset['close'].pct_change()
    direction = dataset['close'].pct_change().shift(-1)
    direction[direction.between(-0.002, 0.002)] = 0
    direction[direction > 0] = 1
    direction[direction < 0] = -1
    dataset['direction'] = direction
    dataset = loc_feature(dataset)
    
    dataset['ema5'] = ta.ema(dataset['close'], length = 6)
    dataset['roc5'] = ta.roc(dataset['close'], length = 6)
    dataset['cci5'] = ta.cci(dataset['high'], dataset['low'],
                                        dataset['close'], length = 6)
    dataset['eom5'] = ta.eom(dataset['high'], dataset['low'],
                                        dataset['close'], dataset['volumes'],
                                        length = 6)
    rolling_window = dataset['returns'].rolling(window = 18, min_periods = 18)
    dataset['skewness'] = rolling_window.skew()
    
    rolling_window_percentiles = 18
    a = dataset["close"].rolling(window = rolling_window_percentiles, min_periods = 18).rank()
    b = dataset["close"].rolling(window = rolling_window_percentiles, min_periods = 18).count()
    percentiles = a/b
    dataset['percentiles'] = percentiles
    dataset.dropna(inplace = True)
    dataset['filtered percentiles'] = savgol_filter(dataset['percentiles'], rolling_window_percentiles, 1)
    dataset.drop(columns = ['percentiles'], inplace = True)
    return dataset
dataset = get_features(ohlc)
dataset.to_csv("dataset.csv")
#%%
def get_corr_matrix(data):
    if data.shape[1] < 14:
        raise ValueError("DataFrame does not have enough columns.")
    selected_columns = [5, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]
    data_subset = data.iloc[:, selected_columns]
    corr_matrix, _ = spearmanr(data_subset)
    correlation_df = pd.DataFrame(corr_matrix, columns=data_subset.columns, index=data_subset.columns)
    return correlation_df
dataset = dataset.reset_index(drop=True)
correlation_df = get_corr_matrix(dataset)
pic_file = "Spearman_Correlation_Heatmap.png"
plt.figure(figsize=(16, 10), dpi=300)
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f', linewidths=.5)
plt.title("Spearman Correlation Heatmap")
plt.xticks(rotation=45)
plt.savefig(pic_file)
plt.show()
# %%

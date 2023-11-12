#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df['Prediction'] = df['Prediction'].replace({0: -1, 1: 0, 2: 1})
    df['Daily_Return'] = df['Prediction'] * df['Returns']
    return df
path_to_files = "../prediction_results/"
files = os.listdir(path_to_files)
rnn_dfs = []
transformer_dfs = []

for file in files:
    if 'rnn' in file:
        rnn_dfs.append(load_and_preprocess(path_to_files + file))
    elif 'transformer' in file:
        transformer_dfs.append(load_and_preprocess(path_to_files + file))

rnn_data = pd.concat(rnn_dfs)
transformer_data = pd.concat(transformer_dfs)
def annualized_return(returns):
    return np.prod(1 + returns)**(252/len(returns)) - 1

def annualized_volatility(returns):
    return np.std(returns) * np.sqrt(252)

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def sharpe_ratio(returns):
    return annualized_return(returns) / annualized_volatility(returns)

def apply_strategy(strategy, data):
    if strategy == "all_long":
        data['Strategy_Return'] = data['Returns']
    elif strategy == "prediction_long_only":
        data['Strategy_Return'] = np.where(data['Prediction'] == 1, data['Returns'], 0)
    elif strategy == "50_long_50_short":
        data['Strategy_Return'] = np.where(data['Prediction'] == 1, 0.5 * data['Returns'], 
                                           np.where(data['Prediction'] == -1, -0.5 * data['Returns'], 0))
    elif strategy == "70_long_30_short":
        data['Strategy_Return'] = np.where(data['Prediction'] == 1, 0.7 * data['Returns'], 
                                           np.where(data['Prediction'] == -1, -0.3 * data['Returns'], 0))
    elif strategy == "90_long_10_short":
        data['Strategy_Return'] = np.where(data['Prediction'] == 1, 0.9 * data['Returns'], 
                                           np.where(data['Prediction'] == -1, -0.1 * data['Returns'], 0))
    return data

strategies = ["all_long", "prediction_long_only", "50_long_50_short", "70_long_30_short", "90_long_10_short"] # 完成策略列表
results = []

for strategy in strategies:
    rnn_strategy_data = apply_strategy(strategy, rnn_data.copy())
    transformer_strategy_data = apply_strategy(strategy, transformer_data.copy())

    for data, model_type in zip([rnn_strategy_data, transformer_strategy_data], ['RNN', 'Transformer']):
        daily_returns = data['Strategy_Return']
        ann_return = annualized_return(daily_returns)
        ann_vol = annualized_volatility(daily_returns)
        max_dd = max_drawdown(daily_returns)
        s_ratio = sharpe_ratio(daily_returns)

        results.append({
            'Strategy': strategy,
            'Model': model_type,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Max Drawdown': max_dd,
            'Sharpe Ratio': s_ratio
        })
def cumulative_return(returns):
    return (1 + returns).cumprod() - 1
cumulative_returns_df = pd.DataFrame()
for data in [rnn_data, transformer_data]:
    for strategy in strategies:
        strategy_data = apply_strategy(strategy, data.copy())
        strategy_data = strategy_data.sort_values(by="Date")
        strategy_data.index = pd.to_datetime(strategy_data['Date'])
        strategy_data['Cumulative Return'] = cumulative_return(strategy_data['Strategy_Return'])
        cumulative_returns_df[strategy] = strategy_data['Cumulative Return']
    plt.figure(figsize=(16, 8), dpi = 400)
    for strategy in strategies:
        plt.plot(cumulative_returns_df[strategy], label=strategy)
    plt.title('Cumulative Returns by Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()
#%%
results_df = pd.DataFrame(results)
print(results_df)
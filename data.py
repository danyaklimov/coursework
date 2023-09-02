import pandas as pd
import numpy as np


def daily_return(df, index):
    daily_return = np.log(df[index] / df[index - 1])
    return daily_return


def read_data() -> tuple:
    """
    читает данные
    :return: кортеж из pd.Series , в каждой Adj_close для каждой компании
    """
    dataset_path_A = '/home/danila/stock_market_dataset/A.csv'
    dataset_path_AA = '/home/danila/stock_market_dataset/AA.csv'
    dataset_path_AAME = '/home/danila/stock_market_dataset/AAME.csv'
    dataset_path_AAL = '/home/danila/stock_market_dataset/AAL.csv'
    dataset_path_AAMC = '/home/danila/stock_market_dataset/AAMC.csv'

    data_A = pd.read_csv(
        dataset_path_A,
        parse_dates=['Date'],
        dayfirst=True,
        keep_default_na=False,
        low_memory=False,
        index_col='Date'
    )
    data_AA = pd.read_csv(
        dataset_path_AA, parse_dates=['Date'], dayfirst=True,
        keep_default_na=False,
        low_memory=False,
        index_col='Date'
    )
    data_AAME = pd.read_csv(
        dataset_path_AAME,
        parse_dates=['Date'],
        dayfirst=True,
        keep_default_na=False,
        low_memory=False,
        index_col='Date'
    )
    data_AAL = pd.read_csv(
        dataset_path_AAL,
        parse_dates=['Date'],
        dayfirst=True,
        keep_default_na=False,
        low_memory=False,
        index_col='Date'
    )
    data_AAMC = pd.read_csv(
        dataset_path_AAMC,
        parse_dates=['Date'],
        dayfirst=True, keep_default_na=False,
        low_memory=False, index_col='Date'
    )

    data_A = data_A.loc['2018-07-01':'2019-06-30']['Adj Close']
    data_AA = data_AA.loc['2018-07-01':'2019-06-30']['Adj Close']
    data_AAME = data_AAME.loc['2018-07-01':'2019-06-30']['Adj Close']
    data_AAL = data_AAL.loc['2018-07-01':'2019-06-30']['Adj Close']
    data_AAMC = data_AAMC.loc['2018-07-01':'2019-06-30']['Adj Close']

    return data_A, data_AA, data_AAME, data_AAL, data_AAMC


def daily(dataframes: tuple) -> list[list]:
    """
    Считает daily_returns для каждой компании
    :return: список со списками из daily returns для каждой компании
    """
    daily_returns = []

    for i in range(len(dataframes)):
        daily_returns_ = []
        for j in range(1, len(dataframes[i])):
            daily_returns_.append(daily_return(dataframes[i], j))
        daily_returns.append(daily_returns_)

    # добавляю ноль в начало daily returns для каждой компании
    for i in range(len(daily_returns)):
        daily_returns[i].insert(0, 0)

    return daily_returns


def daily_returns_dataframe(
        daily_returns: list[list],
        dataframes: tuple
) -> pd.DataFrame:
    # добавляю daily returns в таблицу
    for i in range(len(dataframes)):
        dataframes[i]['daily_returns'] = daily_returns[i]

    frames = [x.daily_returns for x in dataframes]
    result = pd.DataFrame(frames, index=['A', 'AA', 'AAME', 'AAL', 'AAMC'])
    return result

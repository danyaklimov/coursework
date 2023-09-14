import glob

import pandas as pd
import numpy as np


def get_indexes(file_path: str, i) -> list[str]:
    import os
    import re

    file_names_list: list[str] = os.listdir(file_path)
    file_names_str = ' '.join(file_names_list)
    index = re.findall('[A-Z]+', file_names_str)
    return index[i]


def daily_return(df, index):
    daily_return = np.log(df[index] / df[index - 1])
    return daily_return


def read_data(folder_path: str, n_companies) -> list[tuple]:
    """
    Читает данные
    :return: спиок из (pd. Series, index) , в каждой Adj_close для каждой компании
    """

    result = []

    file_list = glob.glob(folder_path + "/*.csv")
    i = 0
    n = 0
    while n != n_companies:
        data = pd.read_csv(
            file_list[i],
            parse_dates=['Date'],
            dayfirst=True,
            keep_default_na=False,
            low_memory=False,
            index_col='Date',
            # on_bad_lines='skip'
        )
        data['Adjusted Close'].replace('', 1, inplace=True)
        # data.dropna(subset=['Adjusted Close'], inplace=True)
        data = data.loc['2021-07-01':'2022-06-30']['Adjusted Close']
        data = data.astype('float')
        if len(data) != 252:
            print("Недостаточно данных")
        elif get_indexes(folder_path, i) in ['GMAN']:
            print("Дерьмовые данные")
        else:
            n += 1
            index = get_indexes(folder_path, i)
            result.append((index, data))
        i += 1

    return result


def daily(dataframes: list) -> list[list]:
    """
    Считает daily_returns для каждой компании
    :input: read_data(folder_path, n_companies)
    :return: список с кортежами (daily returns, индекса) для каждой компании
    """
    daily_returns = []

    for i in range(len(dataframes)):
        daily_returns_ = []
        for j in range(1, len(dataframes[i][1])):
            daily_returns_.append(daily_return(dataframes[i][1], j))
        daily_returns.append(daily_returns_)

    # добавляю ноль в начало daily returns для каждой компании
    for i in range(len(daily_returns)):
        daily_returns[i].insert(0, 0)

    return daily_returns


def daily_returns_dataframe(
        daily_returns: list[list],
        dataframes: list[tuple]
) -> pd.DataFrame:
    """

    :param daily_returns: input daily_returns() result
    :param dataframes:  input read_data() result
    :return: pd.DataFrame
    """
    index = []
    for i in range(len(dataframes)):
        dataframes[i][1]['daily_returns'] = daily_returns[i]
        index.append(dataframes[i][0])

    frames = [x[1].daily_returns for x in dataframes]
    result = pd.DataFrame(frames, index=index)
    return result


if __name__ == '__main__':
    data = read_data(
            '/home/danila/Downloads/archive/stock_market_data/nasdaq/csv',
            10
        )
    daily_returns = daily(data)
    dataframe = daily_returns_dataframe(daily_returns, data)
    print(
        data,
        daily_returns,
        dataframe
    )

import glob
from typing import Any

import pandas as pd
import numpy as np


def get_transformed_data(folder_path: str, start_datetime,
                         stop_datetime) -> pd.DataFrame:
    """
    makes all the transfroms and returns dataframe with daily returns indexed by their tickers
    """
    adj_close_array_and_ticker, _ = read_data(folder_path,
                                              start_datetime, stop_datetime)
    daily_returns = get_daily_returns(adj_close_array_and_ticker)
    dataframe_of_daily_returns_with_ticker = get_dataframe_of_daily_returns_with_ticker_index(
        daily_returns, adj_close_array_and_ticker)

    return dataframe_of_daily_returns_with_ticker


def get_indexes(file_path: str, i) -> list[str]:
    import os
    import re

    file_names_list: list[str] = os.listdir(file_path)
    file_names_str = ' '.join(file_names_list)
    index = re.findall('[A-Z]+', file_names_str)
    return index[i]


def get_index(file_path: str) -> str:
    import re

    index = re.findall('[A-Z]+', file_path)
    return index[1]


def calculate_daily_return(adj_close_array, index):
    daily_return = np.log(adj_close_array[index] / adj_close_array[index - 1])
    return daily_return


def read_data(folder_path: str, start_datetime, stop_datetime
              ) -> tuple[list[tuple[str, pd.Series]], int]:
    """
    Читает данные
    :return: спиок из (pd. Series, index) , в каждой Adj_close для каждой компании
    """

    result = []

    file_list = glob.glob(folder_path + "/*.csv")
    n_companies = len(file_list)
    # companies = []

    i = 0
    n = 0
    while n != n_companies:
        data = pd.read_csv(
            file_list[i],
            # companies[i],
            parse_dates=['Date'],
            dayfirst=True,
            keep_default_na=False,
            low_memory=False,
            index_col='Date',
            # on_bad_lines='skip'
        )
        # data['Adjusted Close'].replace('', 1, inplace=True)
        # data.dropna(subset=['Adjusted Close'], inplace=True)

        # '2021-07-01':'2022-06-30'
        data = data.loc[start_datetime:stop_datetime]['Adj Close']
        # data = data.astype('float')
        # if len(data) != 252:
        #     print("Недостаточно данных")
        # if get_indexes(folder_path, i) in ['GMAN']:
        #     print("Дерьмовые данные")
        # else:
        #     n += 1
        #     index = get_indexes(folder_path, i)
        #     result.append((index, data))

        n += 1
        index = get_index(file_list[i])
        result.append((index, data))
        i += 1

    return result, len(data)


def get_daily_returns(adj_close_array_and_ticker: list) -> list[list]:
    """
    Считает daily_returns для каждой компании
    :input: read_data(folder_path, n_companies)
    :return: список с daily returns для каждой компании
    """
    result = []

    for i in range(len(adj_close_array_and_ticker)):
        _daily_returns = []
        for j in range(1, len(adj_close_array_and_ticker[i][1])):
            _daily_returns.append(
                calculate_daily_return(adj_close_array_and_ticker[i][1], j))
        result.append(_daily_returns)

    # добавляю ноль в начало daily returns для каждой компании
    # for i in range(len(result)):
    #     result[i].insert(0, 0)

    return result


def get_dataframe_of_daily_returns_with_ticker_index(
        daily_returns: list[list],
        adj_close_array_and_ticker: list[tuple]
) -> pd.DataFrame:
    """

    :param daily_returns: result of daily_returns()
    :param adj_close_array_and_ticker:  result of read_data()
    :return: pd.DataFrame
    """
    index = []
    for i in range(len(adj_close_array_and_ticker)):
        # adj_close_array_and_ticker[i][1]['daily_returns'] = daily_returns[i]
        index.append(adj_close_array_and_ticker[i][0])

    # frames = [x[1].daily_returns for x in adj_close_array_and_ticker]
    result = pd.DataFrame(daily_returns, index=index)
    return result


def get_clique(nodes, edges) -> list:
    nodes = list(nodes)
    edges = list(edges)
    clique_list = []

    counter = 0
    for node in nodes:
        if all([True if node in edge else False for edge in edges]):
            clique_list.append(node)
            counter += 1

    if counter < 2:
        return []
    return clique_list


def get_independent_set(nodes, edges) -> list:
    nodes = list(nodes)
    edges = list(edges)
    result = []

    for node in nodes:
        if not any([True if node in edge else False for edge in edges]):
            result.append(node)

    return result


if __name__ == '__main__':
    from datetime import date

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)

    # adj_close_array_and_ticker, _ = read_data(
    #         '/home/danila/Downloads/historical_stock_data',
    #         40,
    #         START,
    #         STOP
    #     )
    # daily_returns = get_daily_returns(adj_close_array_and_ticker)
    # dataframe = get_dataframe_of_daily_returns_with_ticker_index(
    #     daily_returns, adj_close_array_and_ticker)
    # print(dataframe)
    # print(adj_close_array_and_ticker)
    # for i in range(len(adj_close_array_and_ticker)):
    #     for x in adj_close_array_and_ticker[i][1]:
    #         if isinstance(x, str):
    #             print(f'there is str in {adj_close_array_and_ticker[i][0]}')
    #             print(x)

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )
    print(dataframe)

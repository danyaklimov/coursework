import glob

import pandas as pd
import numpy as np


def get_indexes(file_path: str) -> list[str]:
    file_names_list: list[str] = os.listdir(file_path)
    file_names_str = ' '.join(file_names_list)
    index = re.findall('[A-Z]+', file_names_str)
    return index


def daily_return(df, index):
    daily_return = np.log(df[index] / df[index - 1])
    return daily_return


def read_data(folder_path: str) -> list:
    """
    Читает данные
    :return: кортеж из pd. Series , в каждой Adj_close для каждой компании
    """

    result = []

    file_list = glob.glob(folder_path + "/*.csv")
    main_dataframe = pd.DataFrame(pd.read_csv(file_list[0]))
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        data = data.loc['2018-07-01':'2019-06-30']['Adjusted Close']
        result.append(data)

    return result

def daily(dataframes: list) -> list[list]:
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
        dataframes: tuple,
        index: list[str]
) -> pd.DataFrame:
    # добавляю daily returns в таблицу
    for i in range(len(dataframes)):
        dataframes[i]['daily_returns'] = daily_returns[i]

    frames = [x.daily_returns for x in dataframes]
    result = pd.DataFrame(frames, index=index)
    return result

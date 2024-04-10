import numpy as np
import scipy.stats as sps
import pandas as pd


def kurtosis(df, n, N):
    df = np.array(df)
    cov_matrix = np.cov(df)
    sum_ = 0
    for i in range(len(df)):
        sum_ += ((df[:, i] - np.mean(df, axis=1)).dot(
            np.linalg.inv(cov_matrix)).dot(
            (df[:, i] - np.mean(df, axis=1)).T)) ** 2
    result = sum_ * (1 / n) * (1 / (N * (N + 2))) - 1
    return result

def get_threshold_pearson_from_sign(threshold_sign: float) -> float:
    res = -1 * np.cos(np.pi * threshold_sign)
    return res

def get_threshold_pearson_from_kendall(threshold_kendall: float) -> float:
    res = np.sin(np.pi * threshold_kendall / 2)
    return res

def get_t_stat_pearson(threshold, corr_coef, n) -> float:
    # корень из n как в книге колданова
    stat = np.sqrt(n) * (
            0.5 * np.log((1 + corr_coef) / (1 - corr_coef)) - 0.5 * np.log(
        (1 + threshold) / (1 - threshold)))
    return stat

def get_t_stats_pearson_array(threshold, N_companies, daily_returns) -> list[list[float]]:
    # corr_matrix = pd.DataFrame(dict(dataframes)).corr()
    corr_matrix = np.corrcoef(daily_returns)
    # print(corr_matrix)
    t_stats = []

    for i in range(N_companies):
        st = []
        for j in range(N_companies):
            if i == j:
                st.append(0)
            else:
                st.append(get_t_stat_pearson(threshold, corr_matrix[i][j], N_companies))
        t_stats.append(st)

    return t_stats

def get_p_values(
        threshold: float,
        N_days,
        N_companies: int,
        daily_returns: list[list],
        dataframe: pd.DataFrame,
        elliptical=True
) -> list[list[float]]:

    # для нормально распределенного вектора Х
    t_stats = get_t_stats_pearson_array(threshold, N_companies, dataframe)

    k = kurtosis(dataframe, N_days, N_companies)

    p_values_pearson = []

    for i in range(N_companies):
        ps_pearson = []
        for j in range(N_companies):
            if i == j:
                ps_pearson.append(0)
            else:
                if elliptical:
                    ps_pearson.append(sps.norm.cdf(
                        np.sqrt(N_days / (k + 1)) * t_stats[i][j]))
                else:
                    ps_pearson.append(sps.norm.cdf(t_stats[i][j]))

        p_values_pearson.append(ps_pearson)

    return p_values_pearson


if __name__ == '__main__':
    from datetime import date
    from data import get_transformed_data, get_all_indexes

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)
    N_DAYS = (STOP - START).days
    THRESHOLD = 0.3
    NODES = get_all_indexes('/home/danila/Downloads/historical_stock_data', )

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )
    N_COMPANIES = len(dataframe)

    t_stats = get_t_stats_pearson_array(THRESHOLD, N_COMPANIES, dataframe)
    p_values = get_p_values(THRESHOLD, N_DAYS, N_COMPANIES, dataframe, dataframe, False)
    print(np.array(p_values))

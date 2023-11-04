import pandas as pd
import scipy.stats as sps
import numpy as np


def threshold_sign(threshold_kd: float) -> float:
    res = (threshold_kd + 1) / 2
    return res


def threshold_pearson(threshold_sign: float) -> float:
    res = -1 * np.cos(np.pi * threshold_sign)
    return res


def t_stat_pearson(threshold, corr_coef, n):
    # корень из n как в кинге колданова
    stat = np.sqrt(n) * (
            0.5 * np.log((1 + corr_coef) / (1 - corr_coef)) - 0.5 * np.log(
        (1 + threshold) / (1 - threshold)))
    return stat


def t_stat_sign(df, num0, num1):
    def sign(vector0, vector1, mean0, mean1):
        if (vector0 - mean0) * (vector1 - mean1) >= 0:
            return 1
        else:
            return 0

    sum_ = 0
    x0 = df.iloc[num0]
    x1 = df.iloc[num1]
    mean0 = np.mean(x0)
    mean1 = np.mean(x1)

    for i in range(len(x0)):
        sum_ += sign(x0[i], x1[i], mean0, mean1)

    return sum_


def gamma_kendall(
        daily_returns_i: list[float],
        daily_returns_j: list[float],
        n_days: int
) -> float:
    def sign_kd(daily_i_0, daily_i_1, daily_j_0, daily_j_1):
        if (daily_i_0 - daily_i_1) * (daily_j_0 - daily_j_1) >= 0:
            return 1
        else:
            return 0

    gamma_kd = 0

    for i in range(len(daily_returns_i)):
        for j in range(len(daily_returns_j)):
            if i != j:
                gamma_kd += sign_kd(daily_returns_i[i], daily_returns_i[j],
                                    daily_returns_j[i],
                                    daily_returns_j[j])

    gamma_kd = gamma_kd / (n_days * (n_days - 1))

    return gamma_kd


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


def t_stats_pearson_array(threshold, N_companies, daily_returns):
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
                st.append(t_stat_pearson(threshold, corr_matrix[i][j], N_companies))
        t_stats.append(st)

    return t_stats


def t_stats_sign_array(dataframe: pd.DataFrame, N_companies: int):
    t_sign_stats = []

    for i in range(N_companies):
        st = []
        for j in range(N_companies):
            if i == j:
                st.append(0)
            else:
                st.append(t_stat_sign(dataframe, i, j))
        t_sign_stats.append(st)

    return t_sign_stats


def gamma_kendall_array(daily_returns: list[list], N_companies: int) -> list[
    list]:
    gamma_kendall_stats = []

    for i in range(N_companies):
        st = []
        for j in range(N_companies):
            if i == j:
                st.append(
                    0)  # не помню почему заполнял нулями, наверно чтобы квадратная матрица получилась
            else:
                st.append(
                    gamma_kendall(daily_returns[i], daily_returns[j], 250))
        gamma_kendall_stats.append(st)

    return gamma_kendall_stats


def P_c(gamma_kendall: float) -> float:
    return (gamma_kendall + 1) / 2


def P_cc(daily_i: list[float], daily_j: list[float], N_days: int) -> float:
    # индикатор
    def ind_k(daily_i, daily_j, t, s, q):
        if ((daily_i[t] - daily_i[s]) * (daily_j[t] - daily_j[s]) >= 0) and (
                (daily_i[t] - daily_i[q]) * (daily_j[t] - daily_j[q]) >= 0):
            return 1
        else:
            return 0

    res = 0

    for t in range(N_days):
        for s in range(t, N_days):
            if t != s:
                for q in range(s, N_days):
                    if q != s:
                        res += ind_k(daily_i, daily_j, t, s, q)

    res = res / ((N_days * (N_days - 1) * (N_days - 2)) / 6)
    return res


def t_kendall_norm(
        daily_i: list[float],
        daily_j: list[float],
        threshold: float,
        n_days: int
) -> float:
    gamma_kd = gamma_kendall(daily_i, daily_j, n_days)
    p_c = P_c(gamma_kd)
    p_cc = P_cc(daily_i, daily_j, n_days)
    # print('P_cc:', p_cc)
    # print('P_c:', p_c)

    res = (np.sqrt(n_days) * (gamma_kd - threshold)) / (
            4 * np.sqrt(np.abs(p_cc - p_c ** 2)))

    return res


def t_kendall_norm_array(daily_returns: list[list], n_companies: int,
                         n_days: int, threshold: float) -> list[list]:
    t_kendall_array = []

    for i in range(n_companies):
        st = []
        for j in range(n_companies):
            if i == j:
                st.append(0)
            else:
                st.append(t_kendall_norm(daily_returns[i], daily_returns[j],
                                         threshold, n_days))
        t_kendall_array.append(st)

    return t_kendall_array


def p_values(start, stop, folder_path: str,
             threshold: float, N_days, N_companies: int,
             daily_returns: list[list],
             dataframe: pd.DataFrame, elliptical=True):
    threshold_sg: float = threshold_sign(threshold)
    threshold_p: float = threshold_pearson(threshold_sg)

    # для нормально распределенного вектора Х
    t_stats = t_stats_pearson_array(
        threshold_p, N_companies, folder_path, start, stop)
    t_sign_stats = t_stats_sign_array(dataframe, N_companies)
    t_kendall_stats = t_kendall_norm_array(daily_returns, N_companies, N_days,
                                           threshold)

    k = kurtosis(dataframe, N_days, N_companies)

    p_values_pearson = []
    p_values_sign = []
    p_values_kendall = []

    for i in range(N_companies):
        ps_pearson = []
        ps_sign = []
        ps_kendall = []

        for j in range(N_companies):
            if i == j:
                ps_pearson.append(0)
                ps_sign.append(0)
                ps_kendall.append(0)
            else:
                if elliptical:
                    ps_pearson.append(1 - sps.norm.cdf(
                        np.sqrt(N_days / (k + 1)) * t_stats[i][j]))
                    ps_sign.append(1 - sps.norm.cdf(
                        (t_sign_stats[i][j] - N_days * threshold_sg) / (
                            np.sqrt(
                                N_days * threshold_sg * (1 - threshold_sg)))))
                    ps_kendall.append(1 - sps.norm.cdf(t_kendall_stats[i][j]))
                else:
                    ps_pearson.append(1 - sps.norm.cdf(t_stats[i][j]))
                    ps_sign.append(
                        1 - sps.binom.cdf(t_sign_stats[i][j], N_days,
                                          threshold_sg))
        p_values_pearson.append(ps_pearson)
        p_values_sign.append(ps_sign)
        p_values_kendall.append(ps_kendall)

    return p_values_pearson, p_values_sign, p_values_kendall


def edges(p_values_pearson, p_values_sign, p_values_kendall, N_COMPANIES,
          ALPHA, labels, elliptical=True):
    edges_pearson = []
    edges_sign = []
    edges_kendall = []

    for i in range(N_COMPANIES):
        k = 1
        for j in range(k, N_COMPANIES):
            if i != j:
                if p_values_pearson[i][j] < ALPHA:
                    edges_pearson.append((labels[i], labels[j]))
                if p_values_sign[i][j] < ALPHA:
                    edges_sign.append((labels[i], labels[j]))
                if elliptical:
                    if p_values_kendall[i][j] < ALPHA:
                        edges_kendall.append((labels[i], labels[j]))
        k += 1

    if elliptical:
        return edges_pearson, edges_sign, edges_kendall
    else:
        return edges_pearson, edges_sign


if __name__ == '__main__':
    from data import *
    from datetime import date

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        40,
        START,
        STOP
    )

    t_stat_p = t_stats_pearson_array(
        0.1,
        40,
        dataframe
    )

    print(np.array(t_stat_p))

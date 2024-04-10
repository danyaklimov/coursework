import numpy as np
import pandas as pd
import scipy.stats as sps


def _get_gamma_kendall_estimation_for_pair(
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

def get_gamma_kendall_estimations_for_all_pairs(daily_returns: list[list], N_companies: int) -> list[list]:
    gamma_kendall_stats = []

    for i in range(N_companies):
        st = []
        for j in range(N_companies):
            if i == j:
                st.append(
                    0)  # не помню почему заполнял нулями, наверно чтобы квадратная матрица получилась
            else:
                st.append(
                    _get_gamma_kendall_estimation_for_pair(daily_returns[i], daily_returns[j], 250))
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

def _get_kendall_t_stats_for_pair(
        daily_i: list[float],
        daily_j: list[float],
        threshold: float,
        n_days: int
) -> float:
    gamma_kd = _get_gamma_kendall_estimation_for_pair(daily_i, daily_j, n_days)
    p_c = P_c(gamma_kd)
    p_cc = P_cc(daily_i, daily_j, n_days)
    # print('P_cc:', p_cc)
    # print('P_c:', p_c)

    res = (np.sqrt(n_days) * (gamma_kd - threshold)) / (
            4 * np.sqrt(np.abs(p_cc - p_c ** 2)))

    return res


def get_t_stats(
        estimations: np.ndarray, threshold: float, n_days: int) -> np.ndarray:

    t_stats = []

    def formula(estimation: float, threshold: float, n_days: int):
        result = (estimation - threshold) * np.sqrt(((9*n_days*(n_days-1))/(2*(2*n_days+5))))
        return result

    for i in range(len(estimations)):
        st = []
        for j in range(len(estimations)):
            if i == j:
                st.append(0)
            else:
                st.append(formula(estimations[i][j], threshold, n_days))
        t_stats.append(st)

    t_stats = np.array(t_stats)
    return t_stats

def _get_kendall_t_stats(
        estimations: pd.DataFrame,
        threshold: float,
        n_days: int
) -> np.ndarray:

    estimations = estimations.to_numpy()
    t_stats = get_t_stats(estimations, threshold, n_days)

    return t_stats

def get_p_values(
        t_stats: np.ndarray,
        n_companies: int
) -> np.ndarray:

    p_values = []

    for i in range(n_companies):
        ps = []
        for j in range(n_companies):
            if i == j:
                ps.append(None)
            else:
                ps.append(sps.norm.cdf(t_stats[i][j]))
        p_values.append(ps)

    p_values = np.array(p_values)
    return p_values

def get_kendall_t_stats_for_all_pairs(
        daily_returns: list[list],
        n_companies: int,
        n_days: int,
        threshold: float
) -> list[list]:

    t_kendall_array = []

    for i in range(n_companies):
        st = []
        for j in range(n_companies):
            if i == j:
                st.append(0)
            else:
                st.append(_get_kendall_t_stats_for_pair(daily_returns[i], daily_returns[j],
                                                        threshold, n_days))
        t_kendall_array.append(st)

    return t_kendall_array

def get_p_values_for_kendall_network(
        threshold: float,
        N_days,
        N_companies: int,
        daily_returns: list[list]
):
    t_kendall_stats = get_kendall_t_stats_for_all_pairs(daily_returns, N_companies, N_days,
                                                        threshold)
    p_values_kendall = []

    for i in range(N_companies):
        ps_kendall = []

        for j in range(N_companies):
            if i == j:
                ps_kendall.append(0)
            else:
                ps_kendall.append(1 - sps.norm.cdf(t_kendall_stats[i][j]))

        p_values_kendall.append(ps_kendall)

    return p_values_kendall


if __name__ == '__main__':
    from datetime import date
    from data import get_transformed_data, get_all_indexes

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)
    N_DAYS = (STOP - START).days
    THRESHOLD = 0.1
    NODES = get_all_indexes('/home/danila/Downloads/historical_stock_data', )

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )
    N_COMPANIES = len(dataframe)
    # print(dataframe)

    estimations = get_gamma_kendall_estimations_for_all_pairs(
        daily_returns=dataframe,
        N_companies=N_COMPANIES
    )
    print(np.array(estimations))
    print(np.array(estimations).shape)


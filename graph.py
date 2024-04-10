from typing import Tuple, List, Any

import numpy as np
import pandas as pd


def create_edges_from_estimations(
        t_estimates: pd.DataFrame,
        threshold: float,
        nodes: list[str]
) -> list[tuple[str, str]]:
    result = []

    N_COMPANIES = len(t_estimates)

    for i in range(N_COMPANIES):
        k = 1
        for j in range(k, N_COMPANIES):
            if i != j:
                if t_estimates[i][j] <= threshold:
                    result.append((nodes[i], nodes[j]))
        k += 1

    return result


def holm_step_down_procedure(
        p_values: pd.DataFrame,
        alpha: float
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Implements Holm Step Down Procedure
    :param p_values:
    :param alpha:
    :return: indexes of rejected and accepted hypotheses
    """

    N = len(p_values)
    M = N * (N - 1) // 2
    values_with_indices = []

    for i in range(len(p_values)):
        for j in range(i, len(p_values[0])):
            if j != i:
                values_with_indices.append(((i, j), p_values[i][j]))

    sorted_values_with_indices = sorted(values_with_indices,
                                        key=lambda x: x[1])

    rejected_hypotheses = []
    accepted_hypotheses = []

    for k in range(1, M+1):
        min_p_value = sorted_values_with_indices.pop(0)
        if min_p_value[1] >= alpha / (M - k + 1):
            accepted_hypotheses.append(min_p_value[0])
            accepted_hypotheses += [x[0] for x in sorted_values_with_indices]
            return rejected_hypotheses, accepted_hypotheses
        else:
            rejected_hypotheses.append(min_p_value[0])

    return rejected_hypotheses, accepted_hypotheses

def indexes_to_tickers(
        edges_indexes: list[tuple[int, int]],
        list_of_tickers: list[str]
) -> list[tuple[str, str]]:
    """
    Cast edge indexes to their corresponding tickers
    :param edges_indexes:
    :param list_of_tickers:
    :return: List of (ticker_name1, ticker_name2) edges
    """
    edges = []

    for edge in edges_indexes:
        edges.append((list_of_tickers[edge[0]], list_of_tickers[edge[1]]))

    return edges


if __name__ == '__main__':
    from datetime import date
    from data import get_transformed_data, get_all_indexes
    from procedures import TraditionalProcedure, HolmProcedure

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)
    THRESHOLD = 0.3
    NODES = get_all_indexes('/home/danila/Downloads/historical_stock_data', )

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )

    # trad_proc = TraditionalProcedure(dataframe, THRESHOLD, 40)
    # trad_proc.calculate_estimations()
    #
    # print(trad_proc.t_stat_estimations)
    # t_estimations = trad_proc.t_stat_estimations
    #
    # edges = create_edges_from_estimations(t_estimations, THRESHOLD, NODES)
    # print(edges)
    # print(len(edges) == len(set(edges)))

    # =======================================================================

    # Holm Step Down Procedure
    holm_proc = HolmProcedure(dataframe, THRESHOLD, 40)
    holm_proc.calculate_estimations()
    holm_proc.calculate_p_values()
    holm_proc.test_hypotheses()

    # print(holm_proc.t_stat_estimations)
    print(np.array(holm_proc.p_values))
    print(holm_proc.rejected_hypotheses)
    # print(holm_proc.accepted_hypotheses)
    # print(len(holm_proc.rejected_hypotheses)+len(holm_proc.accepted_hypotheses))

    # r, a = holm_step_down_procedure(holm_proc.p_values, holm_proc.alpha)
    # print(r)
    # print(a)
    # print(len(r)+len(a))
    # print(indexes_to_tickers(holm_proc.rejected_hypotheses, NODES))


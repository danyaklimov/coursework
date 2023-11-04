import pandas as pd
from networkx import Graph


def create_edges_from_estimations(
        t_estimates: pd.DataFrame,
        THRESHOLD: float,
        nodes: list[str]
) -> list[tuple[str, str]]:
    result = []

    N_COMPANIES = len(t_estimates)

    for i in range(N_COMPANIES):
        k = 1
        for j in range(k, N_COMPANIES):
            if i != j:
                if t_estimates[i][j] < THRESHOLD:
                    edges_pearson.append((labels[i], labels[j]))
                if p_values_sign[i][j] < ALPHA:
                    edges_sign.append((labels[i], labels[j]))
                if elliptical:
                    if p_values_kendall[i][j] < ALPHA:
                        edges_kendall.append((labels[i], labels[j]))
        k += 1
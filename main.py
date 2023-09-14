import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import re
import networkx as nx
from stats import *
from data import *
import time
import os

THRESHOLDS = [-0.5, -0.1, 0.1, 0.17, 0.2, 0.23, 0.25, 0.3, 0.55, 0.7]
N_COMPANIES = 10
N_DAYS = 30
ALPHA = 0.05
FOLDER_PATH = '/home/danila/Downloads/archive/stock_market_data/nasdaq/csv'

dataframes = read_data(FOLDER_PATH, N_COMPANIES)
daily_returns = np.array(daily(dataframes))
daily_returns_dataframes = daily_returns_dataframe(daily_returns, dataframes)
nodes = [x[0] for x in dataframes]

edges_pearson_list = {}
edges_sign_list = {}
edges_kendall_list = {}

for threshold in THRESHOLDS:
    p_values_pearson, p_values_sign, p_values_kendall = p_values(
        FOLDER_PATH,
        threshold,
        N_DAYS,
        N_COMPANIES,
        daily_returns,
        daily_returns_dataframes
    )
    edges_pearson, edges_sign, edges_kendall = edges(
        p_values_pearson,
        p_values_sign,
        p_values_kendall,
        N_COMPANIES, ALPHA, nodes
    )
    edges_pearson_list[threshold] = edges_pearson
    edges_sign_list[threshold] = edges_sign
    edges_kendall_list[threshold] = edges_kendall

fig, ax = plt.subplots(10, 3, figsize=(10, 40), layout='tight')

if __name__ == '__main__':
    for i, threshold in enumerate(THRESHOLDS):
        G_pearson = nx.Graph()
        G_sign = nx.Graph()
        G_kendall = nx.Graph()

        G_pearson.add_nodes_from(nodes)
        G_sign.add_nodes_from(nodes)
        G_kendall.add_nodes_from(nodes)

        G_pearson.add_edges_from(edges_pearson_list[threshold])
        G_sign.add_edges_from(edges_sign_list[threshold])
        G_kendall.add_edges_from(edges_kendall_list[threshold])

        nx.draw(G_pearson, pos=nx.circular_layout(G_pearson), ax=ax[i, 0],
                with_labels=True)
        ax[i, 0].set_title(f'Pearson, kendall_threshold={threshold}')

        nx.draw(G_sign, pos=nx.circular_layout(G_sign), ax=ax[i, 1],
                with_labels=True)
        ax[i, 1].set_title(f'Sign, kendall_threshold={threshold}')

        nx.draw(G_kendall, pos=nx.circular_layout(G_sign), ax=ax[i, 2],
                with_labels=True)
        ax[i, 2].set_title(f'Kendall, kendall_threshold={threshold}')

    plt.savefig("graph.png")

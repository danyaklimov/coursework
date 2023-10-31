import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import re
import networkx as nx
from stats import *
from data import *
from datetime import date
import time
import os

THRESHOLDS = [0.01, 0.1, 0.17, 0.2, 0.23, 0.25, 0.3, 0.55, 0.6, 0.7]
THRESHOLDS_SIGN = [threshold_sign(t_kd) for t_kd in THRESHOLDS]
THRESHOLDS_PEARSON = [
    round(threshold_pearson(t_sg), 2) for t_sg in THRESHOLDS_SIGN]
N_COMPANIES = 10
# '2021-07-01':'2022-06-30'
START = date(2021, 7, 1)
STOP = date(2021, 7, 20)
ALPHA = 0.05
FOLDER_PATH = '/home/danila/Downloads/archive/stock_market_data/nasdaq/csv'

dataframes, N_DAYS = read_data(FOLDER_PATH, N_COMPANIES, START, STOP)
print('N_DAYS:', N_DAYS)
daily_returns = np.array(daily(dataframes))
daily_returns_dataframes = daily_returns_dataframe(daily_returns, dataframes)
#nodes = [x[0] for x in dataframes]
nodes = [
    'WABC',
    'INFN',
    'MAT',
    'EXPE',
    'PCTY',
    'SPWH',
    'KELYA',
    'EBTC',
    'GTLS',
    'CECO']

edges_pearson_list = {}
edges_sign_list = {}
edges_kendall_list = {}

for threshold in THRESHOLDS:
    p_values_pearson, p_values_sign, p_values_kendall = p_values(
        START,
        STOP,
        FOLDER_PATH,
        threshold,
        N_DAYS,
        N_COMPANIES,
        daily_returns,
        daily_returns_dataframes,
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

table = pd.DataFrame(columns=[
    'threshold',
    'Pearson_Clique',
    'Pearson_Ind_Set',
    'Sign_Clique',
    'Sign_Ind_Set',
    'Kendall_Clique',
    'Kendall_Ind_Set'])

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
        ax[i, 0].set_title(f'Pearson, pearson_threshold={THRESHOLDS_PEARSON[i]}')

        nx.draw(G_sign, pos=nx.circular_layout(G_sign), ax=ax[i, 1],
                with_labels=True)
        ax[i, 1].set_title(f'Sign, sign_threshold={THRESHOLDS_SIGN[i]}')

        nx.draw(G_kendall, pos=nx.circular_layout(G_sign), ax=ax[i, 2],
                with_labels=True)
        ax[i, 2].set_title(f'Kendall, kendall_threshold={threshold}')

        clique_pearson = get_clique(G_pearson.nodes, G_pearson.edges)
        clique_sign = get_clique(G_sign.nodes, G_sign.edges)
        clique_kendall = get_clique(G_kendall.nodes, G_kendall.edges)

        ind_set_pearson = get_independent_set(G_pearson.nodes, G_pearson.edges)
        ind_set_sign = get_independent_set(G_sign.nodes, G_sign.edges)
        ind_set_kendall = get_independent_set(G_kendall.nodes, G_kendall.edges)

        to_merge = pd.DataFrame.from_dict({
                 'threshold': [threshold],
                 'Pearson_Clique': [clique_pearson],
                 'Pearson_Ind_Set': [ind_set_pearson],
                 'Sign_Clique': [clique_sign],
                 'Sign_Ind_Set': [ind_set_sign],
                 'Kendall_Clique': [clique_kendall],
                 'Kendall_Ind_Set': [ind_set_kendall]
             }, )
        table = pd.concat([table, to_merge], ignore_index=True)
        table.reset_index()

    plt.savefig("graph.png")
    table.to_excel('table.xlsx')

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from datetime import date

from clique import percentage_of_reliable_edges
from graph import create_edges_from_estimations, holm_step_down_procedure, \
    indexes_to_tickers

from stats.kendall_statistics import get_p_values, get_t_stats

from data import get_all_indexes, \
    get_transformed_data_column, read_data


class TraditionalProcedure:
    def __init__(self, daily_returns, threshold, n_companies):
        self.estimations_pearson = None
        self.estimations_kendall = None
        self.sample_graph = None
        self.daily_returns = daily_returns
        self.threshold = threshold
        self.N_companies = n_companies
        self.number_of_edges = None

    def calculate_estimations_pearson(self, is_transformed_corr_coef):
        estimations_pearson = self.daily_returns.corr()
        self.estimations_pearson = np.array(estimations_pearson)
        
    def calculate_estimations_kendall(self):
        estimations = self.daily_returns.corr(method='kendall')
        estimations_numpy = np.array(estimations)
        self.estimations_kendall = estimations_numpy

    def construct_sample_graph(self, nodes, estimations):
        edges = create_edges_from_estimations(
            estimations,
            self.threshold,
            nodes
        )
        self.number_of_edges = len(edges)

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        self.sample_graph = graph


class HolmProcedure:
    def __init__(self, daily_returns, threshold, n_companies, alpha=0.01):
        self.edges = None
        self.t_stats = None
        self.rejected_hypotheses = None
        self.accepted_hypotheses = None
        self.sample_graph = None
        self.estimations = None
        self.daily_returns = daily_returns
        self.threshold = threshold
        self.n_companies = n_companies
        self.alpha = alpha
        self.p_values = None
        self.n_days = daily_returns.shape[0]
        self.number_of_edges = None

    def calculate_estimations(self):
        estimations = self.daily_returns.corr(method='kendall')
        estimations_numpy = np.array(estimations)
        self.estimations = estimations_numpy

    def calculate_t_stats(self):
        t_stats = get_t_stats(
            self.estimations,
            self.threshold,
            self.n_days
        )
        self.t_stats = t_stats

    def calculate_p_values(self):
        p_values = get_p_values(
            self.t_stats,
            self.n_companies
        )
        self.p_values = p_values
        
    def test_hypotheses(self):
        rejected, accepted = holm_step_down_procedure(self.p_values, self.alpha)
        self.rejected_hypotheses, self.accepted_hypotheses = rejected, accepted

    def construct_sample_graph(self, nodes):
        edges = indexes_to_tickers(self.rejected_hypotheses, nodes)
        self.number_of_edges = len(edges)
        self.edges = edges

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        self.sample_graph = graph


def main():

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)
    THRESHOLD = 0.2
    NODES = get_all_indexes('/home/danila/Downloads/historical_stock_data')
    _, N_COMPANIES = read_data(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )
    dataframe = get_transformed_data_column(
        '/home/danila/Downloads/historical_stock_data',
        START,
        STOP
    )

    # Traditional Procedure
    procedure = TraditionalProcedure(dataframe, THRESHOLD, N_COMPANIES)
    # procedure.calculate_estimations_pearson(is_transformed_corr_coef=True)
    procedure.calculate_estimations_kendall()
    # procedure.construct_sample_graph(NODES, procedure.estimations_pearson)
    procedure.construct_sample_graph(NODES, procedure.estimations_kendall)

    # =======================================================================

    # Holm Step Down Procedure
    holm_procedure = HolmProcedure(dataframe, THRESHOLD, N_COMPANIES, 0.5)
    holm_procedure.calculate_estimations()
    holm_procedure.calculate_t_stats()
    holm_procedure.calculate_p_values()
    holm_procedure.test_hypotheses()
    holm_procedure.construct_sample_graph(NODES)

    print(holm_procedure.edges)

    # =======================================================================

    # Plotting
    graph = procedure.sample_graph
    # degrees_array = [tup[1] for tup in graph.degree]
    cliques = sorted(list(nx.find_cliques(graph)), key=lambda x: len(x),
                     reverse=True)
    # nx.draw(graph, pos=nx.circular_layout(graph), with_labels=True)
    # plt.savefig("holm_graph.png")

    # plt.hist(degrees_array, bins=60)
    # plt.title('Degree histogram')
    # plt.xlabel('Degree')
    # plt.ylabel('# of Nodes')
    plt.savefig("degree_distr.png")

    # x = [-0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2]
    # pears = [8,	26,	99,	213,	350,	510,	691]
    # kendall = [2,	12,	86,	207,	394,	637,	916]
    # kendall_holm = [0, 0, 0, 1, 10, 60, 122]
    # plt.plot(x, pears, label='Pearson traditional')
    # plt.plot(x, kendall_holm, label='Kendall Holm')
    # plt.xlabel('threshold')
    # plt.ylabel('# of Degrees')
    # plt.legend()
    # plt.savefig('speed')

    # print(f'Cliques: {cliques}')
    # print(f'Clique distr: {[len(x) for x in cliques]}')
    # print(f'Degree distr: {degrees_array}')
    # print(f'List of degrees: {graph.degree}')
    # print(f'Number of edges: {procedure.number_of_edges}')
    # print(f'Percentages of reliable edges: {percentage_of_reliable_edges(holm_procedure.edges, cliques)}')


if __name__ == '__main__':
    main()

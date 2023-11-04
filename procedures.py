from stats import t_stats_pearson_array
import numpy as np

class TraditionalProcedure:
    def __init__(self, daily_returns, threshold, N_companies):
        self.t_stat_estimations = None
        self.daily_returns = daily_returns
        self.threshold = threshold
        self.N_companies = N_companies

    def calculate_estimations(self):
        t_stats_pearson = t_stats_pearson_array(
            self.threshold,
            self.N_companies,
            self.daily_returns
        )
        self.t_stat_estimations = np.array(t_stats_pearson)

    def construct_sample_graph(self):
        pass


class HolmProcedure:
    def calculate_estimations(self):
        pass

    def construct_sample_graph(self):
        pass

if __name__ == '__main__':
    from datetime import date
    from data import get_transformed_data

    START = date(2022, 10, 31)
    STOP = date(2023, 10, 30)
    THRESHOLD = 0.3

    dataframe = get_transformed_data(
        '/home/danila/Downloads/historical_stock_data',
        40,
        START,
        STOP
    )

    trad_proc = TraditionalProcedure(dataframe, THRESHOLD, 40)
    trad_proc.calculate_estimations()

    print(trad_proc.t_stat_estimations)

import numpy as np
import math


def is_sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def percentage_of_reliable_edges(
        reliable_edges: list[tuple[str, str]],
        cliques: list[list[str]]
) -> dict:
    cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
    cliques = list(filter(lambda x: len(x) > 2, cliques))

    percentages = {}

    for x in cliques:
        size = len(x)
        number_of_edges_in_clique = math.factorial(size) / (
                    2 * math.factorial(size - 2))

        if size not in percentages.keys(): percentages[size] = []

        number_of_matched_edges = 0

        for reliable_edge in reliable_edges:
            if is_sublist(list(reliable_edge), x):
                number_of_matched_edges += 1

        percentages[size].append(
            number_of_matched_edges / number_of_edges_in_clique)

    for key in percentages.keys():
        percentages[key] = np.mean(percentages[key])

    return percentages

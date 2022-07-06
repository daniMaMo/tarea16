import csv
import timeit
import numpy as np
import pandas as pd

# Este es la propuesta de pareto hecha por mi parte, usando los conceptos vistos en clase

with open('statistics.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    VECTORS = {(row['Symbol 1'], row['Symbol 2'], float(row['APR']), float(row['SHARPE']),
                float(row['price'])) for row in reader}


def pareto(vectors):
    result = []
    for row in vectors:
        for ROW in vectors - {row}:
            if (ROW[2] <= row[2]) and (ROW[3] <= row[3]) and (ROW[4] <= row[4]):
                result.append(ROW)
    return result


DOMINADOS_SET = set(pareto(VECTORS))
PARETO = VECTORS - DOMINADOS_SET
# print('El resultado usado pareto es:', PARETO)


with open('pareto.csv', mode='w') as csv_file:
    fieldnames = ['Symbol 1', 'Symbol 2', 'APR', 'SHARPE', 'price']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    fields = ['Symbol 1', 'Symbol 2', 'APR', 'SHARPE', 'price']
    for row in PARETO:
        values = [row[0], row[1], row[2], row[3], row[4]]
        row_dict = dict(zip(fields, values))
        writer.writerow(row_dict)

# Aquí mostraré los métodos propuestos en clase

DATA = pd.read_csv('statistics.csv', sep=',', index_col=['Symbol 1',
                                                         'Symbol 2'],
                   usecols=['Symbol 1', 'Symbol 2', 'APR', 'SHARPE', 'price'])


# Very slow for many datapoints.  Fastest for many costs, most readable

def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient


# answer = is_pareto_efficient_dumb(-DATA.values)
# indices = list(np.nonzero(answer)[0].tolist())
# PARETO_EFFICIENT_DUMB = DATA.iloc[indices]
# print(PARETO_EFFICIENT_DUMB)

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# answer_pareto_simple = is_pareto_efficient_simple(-DATA.values)
# indices_simple = list(np.nonzero(answer_pareto_simple)[0].tolist())
# PARETO_EFFICIENT_SIMPLE = DATA.iloc[indices_simple]
# print(PARETO_EFFICIENT_SIMPLE)

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


# answer_pareto_efficient = is_pareto_efficient_simple(-DATA.values)
# indices_efficient = list(np.nonzero(answer_pareto_efficient)[0].tolist())
# PARETO_EFFICIENT = DATA.iloc[indices_simple]
# print(PARETO_EFFICIENT)


print('El promedio de tiempo para pareto es:', (timeit.timeit('pareto(VECTORS)', number=10, globals=globals())) / 10)
print('El promedio de tiempo para PARETO_EFFICIENT_DUMB es:',
      (timeit.timeit('is_pareto_efficient_dumb(-DATA.values)', number=10, globals=globals())) / 10)
print('El promedio de tiempo para PARETO_EFFICIENT_SIMPLE es:',
      (timeit.timeit('is_pareto_efficient_simple(-DATA.values)', number=10, globals=globals())) / 10)
print('El promedio de tiempo para PARETO_EFFICIENT es:',
      (timeit.timeit('is_pareto_efficient(-DATA.values)', number=10, globals=globals())) / 10)

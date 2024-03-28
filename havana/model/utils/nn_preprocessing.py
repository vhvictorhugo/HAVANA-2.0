import random

import numpy as np


def one_hot_decoding_predicted(data):
    new = []
    for e in data:
        node_label = []
        for node in e:
            node_label.append(np.argmax(node))
        new.append(node_label)

    new = np.array(new).flatten()
    return new


def top_k_rows(data, k):
    row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i])

    row_sum = sorted(row_sum, reverse=True, key=lambda e: e[0])
    row_sum = row_sum[:k]

    row_sum = [e[1] for e in row_sum]
    random.seed(1)

    return np.array(row_sum)


def split_graph(data, k, split):
    graph = []

    if data.ndim == 2:
        for i in range(1, split + 1):
            matrix = data[k * (i - 1) : k * i, k * (i - 1) : k * i]
            graph.append(matrix)
    else:
        for i in range(1, split + 1):
            matrix = data[k * (i - 1) : k * i]
            graph.append(matrix)

    return np.array(graph)


def top_k_rows_order(graph, k):
    new_graph = []
    # soma dos pesos de todas as arestas do grafo
    matrix_total = np.array(graph).sum()
    for i in range(len(graph)):
        degree = 0
        row = graph[i]
        # Métrica 1: soma dos pesos das arestas do vértice/PoI "i".
        row_total = sum(row)
        # Métrica 1: ponderar os pesos das arestas do vértice/PoI "i" com base no peso total de todas as arestas do grafo.
        row_total = row_total / matrix_total
        for j in range(len(row)):
            # Métrica 2 : contabilizar o grau do vértice/PoI "i".
            if row[j] != 0:
                degree += 1
        # Métrica 2: grau do vértice/PoI "i" é ponderado com base na quantidade de vértices/PoIs do grafo.
        degree = degree / len(graph)

        # Métrica resultante: aplicar as métricas 1 e 2 na fórmula do f1-score.
        new_graph.append([i, (2 * row_total * degree) / (row_total + degree)])

    # Ordenar os PoIs com base na métrica resultante (quanto maior o valor, melhor)
    new_graph = sorted(new_graph, reverse=True, key=lambda e: e[1])
    new_graph = [i[0] for i in new_graph]
    # retorna o grafo com os Top k PoIs.
    new_graph = new_graph[:k]

    return np.array(new_graph)

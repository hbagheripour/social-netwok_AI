# Social Network Chapter 6
# Haydeh Bagheripour

# Problems:
# In this problem, there are two networks: one is the observed network, i.e, the edge
# between P2P clients and the other is the hierarchical tree structure that is used to
# generated the edges in the observed network.
# For this exercise, we will use a complete, perfectly balanced b-ary tree T (each
# node has b children and b ≥ 2), and a network whose nodes are the leaves of T . For
# any pair of network nodes v and w, h(v, w) denotes the distance between the nodes
# and is defined as the height of the subtree L(v, w) of T rooted at the lowest common
# ancestor of v and w. The distance captures the intuition that clients in the same city
# are more likely to be connected than, for example, in the same state.
# To model this intuition, generate a random network on the leaf nodes where for a
# node v, the probability distribution of node v creating an edge to any other node w
# is given by Eq. 6.1


# Ex 46:  "Create random networks for α = 0.1, 0.2, . . . , 10. For each of these networks,
# sample 1000 unique random (s, t) pairs (s = t). Then do a decentralized search
# starting from s as follows. Assuming that the current node is s, pick its neighbour
# u with smallest h(u, t) (break ties arbitrarily). If u = t, the search succeeds. If
# h(s, t) > h(u, t), set s to u and repeat. If h(s, t) ≤ h(u, t), the search fails.
# For each α, pick 1000 pairs of nodes and compute the average path length for the
# searches that succeeded. Then draw a plot of the average path length as a function
# of α. Also, plot the search success probability as a function of α."

import networkx as nx
import random
import matplotlib.pyplot as plt


def generate_random_network(height, b, k, alpha):
    T = nx.balanced_tree(b, height)
    network = nx.DiGraph()

    for node in T.nodes():
        p = {v: b ** (-alpha * nx.shortest_path_length(T, source=node, target=v)) for v in T.nodes()}
        Z = sum(p.values())
        for _ in range(k):
            w = random.choices(list(p.keys()), list(p.values()))[0]
            network.add_edge(node, w)
            p[w] = 0
            Z -= p[w]
            if Z > 0:
                p = {v: p[v] / Z for v in p}

    return network


def decentralized_search(network, s, t):
    current_node = s
    path_length = 0

    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        neighbors.sort(key=lambda x: nx.shortest_path_length(network, source=x, target=t))
        next_node = neighbors[0]

        if nx.shortest_path_length(network, source=current_node, target=t) > nx.shortest_path_length(network,
                                                                                                     source=next_node,
                                                                                                     target=t):
            current_node = next_node
            path_length += 1
        else:
            return float('inf')

    return path_length



height = 10
b = 2
k = 5
alphas = [0.1 * i for i in range(11)]
# alphas = [0.1 * i for i in range(2)]
num_pairs = 1000
average_path_lengths, search_success_probabilities = decentralized_search(height, b, k, alphas, num_pairs)

# Plot 
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(alphas, average_path_lengths, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Average Path Length')
plt.title('Average Path Length vs. Alpha')

plt.subplot(1, 2, 2)
plt.plot(alphas, search_success_probabilities, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Search Success Probability')
plt.title('Search Success Probability vs. Alpha')

plt.tight_layout()
plt.show()
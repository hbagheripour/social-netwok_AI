
# Social Network Chapter 5
# Haydeh Bagheripour
# Ex 39-45

# Problems:

# "Download the Friendster undirected social network data available at https://snap.
# stanford.edu/data/com-Friendster.html.
# This network consists of 65 million nodes and 180 million edges. The world’s
# population in 2012 was 7 billion people. This means that the network has 1% of the
# world’s inhabitants.
# For this graph, compute the following network parameters:

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Ex 39 "Degree distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.Graph(), nodetype=int)

num_runs = 10 
degree_dist = {}

for _ in range(num_runs):
    degree_sequence = [degree for node, degree in G.degree()]
    degree_sequence = sorted(degree_sequence, reverse=True)

    for degree in degree_sequence:
        if degree in degree_dist:
            degree_dist[degree] += 1
        else:
            degree_dist[degree] = 1

total_nodes = G.number_of_nodes()
degree_dist_normalized = {degree: count / (num_runs * total_nodes) for degree, count in degree_dist.items()}

# Sort 
sorted_degree_dist = sorted(degree_dist_normalized.items())

# Get the degrees and corresponding fractions
degrees, fractions = zip(*sorted_degree_dist)





# Ex 40 "Path length distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.Graph(), nodetype=int)

num_walks = 1000
max_walk_length = 100

path_lengths = []
for node in G.nodes():
    for _ in range(num_walks):
        current_node = node
        walk_length = 0
        while walk_length < max_walk_length:
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) > 0:
                next_node = np.random.choice(neighbors)
                current_node = next_node
                walk_length += 1
            else:
                break
        path_lengths.append(walk_length)

path_length_dist = {}
total_paths = len(path_lengths)
for length in path_lengths:
    if length in path_length_dist:
        path_length_dist[length] += 1
    else:
        path_length_dist[length] = 1

path_length_dist_normalized = {length: count / total_paths for length, count in path_length_dist.items()}

sorted_path_lengths = sorted(path_length_dist_normalized.items())

lengths, fractions = zip(*sorted_path_lengths)

plt.plot(lengths, fractions, marker='o')
plt.xlabel('Path Length')
plt.ylabel('Fraction')
plt.title('Path Length Distribution')
plt.show()






# Ex 41 "WCC size distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.DiGraph(), nodetype=int)
wcc = nx.weakly_connected_components(G)
wcc_sizes = [len(component) for component in wcc]

wcc_size_count = {}
for size in wcc_sizes:
    if size not in wcc_size_count:
        wcc_size_count[size] = 0
    wcc_size_count[size] += 1


sorted_sizes = sorted(wcc_size_count.keys())

wcc_size_distribution = [wcc_size_count[size] for size in sorted_sizes]







# Ex 42 "Clustering coefficient distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.DiGraph(), nodetype=int)

clustering_coeffs = nx.clustering(G)

# Count the number of nodes for each CC
clustering_coeff_distribution = [clustering_coeff_count[coeff] for coeff in sorted_coeffs]








#Ex 43 "k-core node size distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.Graph(), nodetype=int)

core_numbers = nx.core_number(G)

core_number_count = {}
for number in core_numbers.values():
    if number not in core_number_count:
        core_number_count[number] = 0
    core_number_count[number] += 1

# Sort 
sorted_numbers = sorted(core_number_count.keys())

# Count the number of nodes for each core number
core_number_distribution = [core_number_count[number] for number in sorted_numbers]







# Ex 44 "Average friends-of-friends distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.Graph(), nodetype=int)

avg_friends_of_friends = {}
for node in G.nodes():
    friends_of_friends = set()
    for neighbor in G.neighbors(node):
        friends_of_friends.update(G.neighbors(neighbor))
    friends_of_friends.discard(node)
    avg_friends_of_friends[node] = len(friends_of_friends) / len(list(G.neighbors(node))) if len(list(G.neighbors(node))) > 0 else 0


friends_of_friends_count = {}
for value in avg_friends_of_friends.values():
    if value not in friends_of_friends_count:
        friends_of_friends_count[value] = 0
    friends_of_friends_count[value] += 1

sorted_values = sorted(friends_of_friends_count.keys())

friends_of_friends_distribution = [friends_of_friends_count[value] for value in sorted_values]








# Ex 45 "Average neighbour degree distribution"

G = nx.read_edgelist('soc-Epinions1.txt', comments='#', delimiter='\t', create_using=nx.Graph(), nodetype=int)

avg_neighbor_degree = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    neighbor_degrees = [G.degree(neighbor) for neighbor in neighbors]
    avg_neighbor_degree[node] = sum(neighbor_degrees) / len(neighbors) if len(neighbors) > 0 else 0


neighbor_degree_count = {}
for value in avg_neighbor_degree.values():
    if value not in neighbor_degree_count:
        neighbor_degree_count[value] = 0
    neighbor_degree_count[value] += 1


sorted_values = sorted(neighbor_degree_count.keys())


neighbor_degree_distribution = [neighbor_degree_count[value] for value in sorted_values]


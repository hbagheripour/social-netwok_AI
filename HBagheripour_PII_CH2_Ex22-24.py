# Social Network Chapter 2
# Haydeh Bagheripour
# Ex 22-24

# Problems:
# Download the Epinions directed network from the SNAP dataset repository available at http://snap.stanford.edu/data/soc-Epinions1.html.
# For this dataset compute the structure of this social network using the same methods as Broder et al. employed.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Ex 22 "Compute the in-degree and out-degree distributions and plot the power law for each of these distributions."

dataset = np.loadtxt("soc-Epinions1.txt", delimiter='\t', skiprows=4)

# Compute in-degree and out-degree
in_degree_counts = {}
out_degree_counts = {}

for edge in dataset:
    source = int(edge[0])
    target = int(edge[1])

    # Update in-degree counts
    if target in in_degree_counts:
        in_degree_counts[target] += 1
    else:
        in_degree_counts[target] = 1

    # Update out-degree counts
    if source in out_degree_counts:
        out_degree_counts[source] += 1
    else:
        out_degree_counts[source] = 1

# Calculate the frequencies of in-degree and out-degree
in_degree_freq = np.array(list(in_degree_counts.values()))
out_degree_freq = np.array(list(out_degree_counts.values()))

# Sort the degree frequencies in descending order
in_degree_freq.sort()
out_degree_freq.sort()

# Define a helper function to fit a line and extract the slope
def fit_power_law(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope


plt.scatter(range(1, len(in_degree_freq) + 1), in_degree_freq, color='b', s=5, label='In-degree')

plt.scatter(range(1, len(out_degree_freq) + 1), out_degree_freq, color='r', s=5, label='Out-degree')

# Fit lines and extract slopes
in_degree_slope = fit_power_law(range(1, len(in_degree_freq) + 1), in_degree_freq)
out_degree_slope = fit_power_law(range(1, len(out_degree_freq) + 1), out_degree_freq)

plt.xlabel('Degree (log scale)')
plt.ylabel('Frequency (log scale)')
plt.title('Power Law Distribution')
plt.legend()

plt.show()





# Ex 23 "Choose 100 nodes at random from the network and do one forward and one backward BFS traversal for each node. Plot the cumulative distributions
# of the nodes covered in these BFS runs as shown in Fig. 2.7. Create one figure for the forward BFS and one for the backward BFS. How many nodes are in the OUT
# and IN components? How many nodes are in the TENDRILS component? (Hint: The forward BFS plot gives the number of nodes in SCC+OUT and similarly,
# the backward BFS plot gives the number of nodes in SCC+IN)."

dataset = np.loadtxt("soc-Epinions1.txt", delimiter='\t', skiprows=4)

adj_list = {}
for edge in dataset:
    source = int(edge[0])
    target = int(edge[1])

    if source in adj_list:
        adj_list[source].append(target)
    else:
        adj_list[source] = [target]

# Randomly select 100 nodes
random_nodes = np.random.choice(list(adj_list.keys()), size=100, replace=False)

# Forward BFS traversal
def forward_bfs(start_node):
    visited = set()
    queue = deque([start_node])
    count = 0

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            count += 1
            if node in adj_list:
                queue.extend(adj_list[node])

    return count

# Backward BFS traversal
def backward_bfs(start_node):
    visited = set()
    queue = deque([start_node])
    count = 0

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            count += 1
            if node in adj_list:
                queue.extend(adj_list[node])

    return count

# Perform forward and backward BFS traversals on selected nodes
forward_bfs_counts = [forward_bfs(node) for node in random_nodes]
backward_bfs_counts = [backward_bfs(node) for node in random_nodes]

# Compute cumulative distributions
forward_bfs_counts.sort()
forward_bfs_cumulative = np.cumsum(forward_bfs_counts) / np.sum(forward_bfs_counts)

backward_bfs_counts.sort()
backward_bfs_cumulative = np.cumsum(backward_bfs_counts) / np.sum(backward_bfs_counts)

# Plot 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(len(forward_bfs_cumulative)), forward_bfs_cumulative)
plt.xlabel('Number of Nodes')
plt.ylabel('Cumulative Distribution')
plt.title('Forward BFS')

plt.subplot(1, 2, 2)
plt.plot(range(len(backward_bfs_cumulative)), backward_bfs_cumulative)
plt.xlabel('Number of Nodes')
plt.ylabel('Cumulative Distribution')
plt.title('Backward BFS')

plt.tight_layout()
plt.show()

# Calculate the number of nodes in OUT, IN, and TENDRILS components
num_nodes_out = forward_bfs_cumulative[-1]
num_nodes_in = backward_bfs_cumulative[-1]
num_nodes_tendrils = len(adj_list) - num_nodes_out - num_nodes_in

print("Nodes in OUT component:", num_nodes_out)
print("Nodes in IN component:", num_nodes_in)
print("Nodes in TENDRILS component:", num_nodes_tendrils)








# Ex 24 "What is the probability that a path exists between two nodes chosen uniformly from the graph? What if the node pairs are only drawn 
# from the WCC of the two networks? Compute the percentage of node pairs that were connected in each of these cases."

dataset = np.loadtxt("soc-Epinions1.txt", delimiter='\t', skiprows=4)

# Create a directed graph
graph = nx.DiGraph()
graph.add_edges_from(dataset)

# Compute the probability of a path existing between two randomly chosen nodes from the graph
num_nodes = len(graph.nodes())
num_trials = 10000
connected_count = 0

for _ in range(num_trials):
    while True:
        node1, node2 = np.random.choice(list(graph.nodes()), size=2, replace=False)
        if graph.out_degree(node1) > 0 and graph.out_degree(node2) > 0:
            break

    if nx.has_path(graph, node1, node2):
        connected_count += 1

probability_graph = connected_count / num_trials

# Compute the probability of a path existing between two randomly chosen nodes from the WCC of the graph
wcc = max(nx.weakly_connected_components(graph), key=len)
wcc_graph = graph.subgraph(wcc)

connected_count_wcc = 0

for _ in range(num_trials):
    while True:
        node1, node2 = np.random.choice(list(wcc), size=2, replace=False)
        if wcc_graph.out_degree(node1) > 0 and wcc_graph.out_degree(node2) > 0:
            break

    if nx.has_path(wcc_graph, node1, node2):
        connected_count_wcc += 1

probability_wcc = connected_count_wcc / num_trials

# Calculate the percentage of node pairs that were connected in each case
percentage_graph = probability_graph * 100
percentage_wcc = probability_wcc * 100

print("Percentage of connected node pairs in the graph:", percentage_graph)
print("Percentage of connected node pairs in the WCC:", percentage_wcc)

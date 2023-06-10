# Social Network Chapter 3 
# Haydeh Bagheripour
# Ex 25-32

import urllib.request
import matplotlib.pyplot as plt
import networkx as nx
import gzip

#Download the Astro Physics collaboration network from the SNAP dataset repository available at http://snap.stanford.edu/data/ca-AstroPh.html. This co-authorship
#network contains 18772 nodes and 198110 edges.
#Generate the graph for this dataset (we will refer to this graph as the real world
#graph).

# Chapter 3 Problems:


# Ex 25 "Erdös–Rényi random graph (G(n, m): Generate a random instance of this model
# by using the number of nodes and edges as the real world graph.""

# Download the Astro Physics collaboration net dataset
url = "http://snap.stanford.edu/data/ca-AstroPh.txt.gz"
f_name = "ca-AstroPh.txt.gz"
urllib.request.urlretrieve(url, f_name)

num_nodes = 18772
num_edges = 198110


# Generate the Erdős-Rényi random graph
G_e_r = nx.gnm_random_graph(num_nodes, num_edges)

print("Number of nodes:", G_e_r.number_of_nodes())
print("Number of edges:", G_e_r.number_of_edges())

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_e_r, seed=42)
nx.draw_networkx(G_e_r, pos, with_labels=False, node_size=10)
plt.title("Erdös-Rényi Random graph(G(n, m)")
plt.axis("off")
plt.show()







# Ex 26 "Configuration model random graph: Generate a random instance of this model
# by using the graph in the dataset."

G_r = nx.read_edgelist(f_name)

# Generate the Configuration model random graph:
degrees = [d for (n, d) in G_r.degree()]
G_conf = nx.configuration_model(degrees)

# Remove parallel edges and self-loops
G_conf= nx.Graph(G_conf)

print("Number of nodes in the real world graph:", G_r.number_of_nodes())
print("Number of edges in the real world graph:", G_r.number_of_edges())

# Print the number of nodes and edges in the configuration model random graph
print("Number of nodes in the Configuration Model random graph:", G_conf.number_of_nodes())
print("Number of edges in the Configuration Model random graph:", G_conf.number_of_edges())

# Draw the real-world graph
plt.figure(figsize=(8, 6))
nx.draw(G_r, node_size=20)
plt.title("Real world graph")
plt.show()

plt.figure(figsize=(8, 6))
nx.draw(G_conf, node_size=20)
plt.title("The Configuration model random graph")
plt.show()







# Ex 27 "Degree distributions"

G = nx.read_edgelist('ca-AstroPh.txt', comments='#', delimiter='\t')

# Compute the degree distribution of the real-world graph
degree_seq = [d for n, d in G.degree()]
hist_g = nx.degree_histogram(G)
plt.bar(range(len(hist_g)), hist_g)
plt.title("Degree distribution of Astro Physics collaboration network")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

# Define the number of nodes and probability for the Erdős-Rényi random graph
num_nodes = 1000
p = 0.01

e_r_graph = nx.erdos_renyi_graph(num_nodes, p)

# Compute the degree distribution of the Erdős-Rényi random graph
degree_seq = [d for n, d in e_r_graph.degree()]
hist_g = nx.degree_histogram(e_r_graph)

# Plot the degree distribution of the Erdős-Rényi random graph
plt.bar(range(len(hist_g)), hist_g)
plt.title("Degree distribution of the Erdős-Rényi random graph")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

# Generate an empty graph with the same degree sequence as the real-world graph
EG = nx.configuration_model(G.degree())

# Remove self-loops and parallel edges from the graph
EG = nx.Graph(EG)

# Compute the degree distribution of the Configuration model random graph 
degree_seq = [d for n, d in EG.degree()]
hist_g= nx.degree_histogram(EG)

plt.bar(range(len(hist_g)), hist_g)
plt.title("Degree distribution of configuration model random graph")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()







# Ex 28 "Shortest path length distributions"
# Load the graph from the file
G = nx.read_edgelist('ca-AstroPh.txt', comments='#', delimiter='\t')

# Compute the shortest path length distribution of the real world graph
s_p_l = dict(nx.shortest_path_length(G))
dist_g = [len([1 for k,v in s_p_l.items() if k!=n and v.get(n) == d]) for n in G.nodes() for d in range(max(s_p_l[n].values())+1)]

# Plot the shortest path length distribution of the real-world graph
plt.bar(range(len(dist_g)), dist_g)
plt.title("Shortest path length distribution")
plt.xlabel("Shortest path length")
plt.ylabel("Frequency")
plt.show()

# Define the number of nodes and probability for the Erdős-Rényi random graph
num_nodes = 1000
p = 0.01

# Generate the Erdős-Rényi random graph
e_r_graph = nx.erdos_renyi_graph(num_nodes, p)

# Compute the shortest path length distribution of the Erdős-Rényi random graph
s_p_l = dict(nx.shortest_path_length(e_r_graph))
dist_g = [len([1 for k,v in s_p_l.items() if k!=n and v.get(n) == d]) for n in e_r_graph.nodes() for d in range(max(s_p_l[n].values())+1)]

# Plot the shortest path length distribution of the Erdős-Rényi random graph
plt.bar(range(len(dist_g)), dist_g)
plt.title("Shortest path length distribution of Erdős-Rényi random graph")
plt.xlabel("Shortest path length")
plt.ylabel("Frequency")
plt.show()

# Generate an empty graph with the same degree sequence as the real-world graph
EG = nx.configuration_model(G.degree())

# Remove self-loops and parallel edges from the graph
EG = nx.Graph(EG)

# Compute the shortest path length distribution of the Configuration model random graph
s_p_l = dict(nx.shortest_path_length(EG))
dist_g = [len([1 for k,v in s_p_l.items() if k!=n and v.get(n) == d]) for n in EG.nodes() for d in range(max(s_p_l[n].values())+1)]

plt.bar(range(len(dist_g)), dist_g)
plt.title("Shortest path length distribution of Configuration model random graph")
plt.xlabel("Shortest path length")
plt.ylabel("Frequency")
plt.show()







# Ex 29 "Clustering coefficient distributions"

Real_G = nx.read_edgelist('ca-AstroPh.txt')

# Generate Erdős-Rényi random graph
n = 18772  
m = 198110  
Erdos_G = nx.gnm_random_graph(n, m)

# Generate Configuration model random graph
degree_seq = [Real_G.degree(node) for node in Real_G.nodes()]
Config_g = nx.configuration_model(degree_seq)

# Compute degree distributions
degree_dist_real = nx.degree_histogram(Real_G)
degree_dist_erdos = nx.degree_histogram(Erdos_G)
degree_dist_config = nx.degree_histogram(Config_g)

# Compute shortest path length distributions
shortest_path_lengths_real = nx.shortest_path_length(Real_G)
path_lengths_real = [length for lengths in shortest_path_lengths_real.values() for length in lengths.values()]
shortest_path_lengths_erdos = nx.shortest_path_length(Erdos_G)
path_lengths_erdos = [length for lengths in shortest_path_lengths_erdos.values() for length in lengths.values()]
shortest_path_lengths_config = nx.shortest_path_length(Config_g)
path_lengths_config = [length for lengths in shortest_path_lengths_config.values() for length in lengths.values()]

# Compute clustering coefficient distributions
clustering_coeffs_real = nx.clustering(Real_G)
clustering_coeffs_values_real = list(clustering_coeffs_real.values())
clustering_coeffs_erdos = nx.clustering(Erdos_G)
clustering_coeffs_values_erdos = list(clustering_coeffs_erdos.values())
clustering_coeffs_config = nx.clustering(Config_g)
clustering_coeffs_values_config = list(clustering_coeffs_config.values())

# plot
plt.plot(degree_dist_real, 'bo-', label='Real World Graph')
plt.plot(degree_dist_erdos, 'ro-', label='Erdos-Renyi Graph')
plt.plot(degree_dist_config, 'go-', label='Configuration Model Graph')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title








# Ex 30 "WCC size distributions"

Real_G = nx.read_edgelist('ca-AstroPh.txt')

# Generate Erdős-Rényi random graph
n = 18772  
m = 198110 
Erdos_G= nx.gnm_random_graph(n, m)

# Generate Configuration model random graph
degree_seq = [Real_G.degree(node) for node in Real_G.nodes()]
Config_g = nx.configuration_model(degree_seq)

# Compute WCC size distributions
wcc_real = nx.weakly_connected_components(Real_G)
wcc_sizes_real = [len(wcc) for wcc in wcc_real]
wcc_erdos = nx.weakly_connected_components(Erdos_G)
wcc_sizes_erdos = [len(wcc) for wcc in wcc_erdos]
wcc_config = nx.weakly_connected_components(Config_g)
wcc_sizes_config = [len(wcc) for wcc in wcc_config]

# Plot WCC size distributions
plt.hist(wcc_sizes_real, bins=range(1, max(wcc_sizes_real) + 2), align='left', rwidth=0.8, label='The real world graph')
plt.xlabel('WCC Size')
plt.ylabel('Count')
plt.title('WCC size distribution of real world graph')
plt.legend()
plt.show()

plt.hist(wcc_sizes_erdos, bins=range(1, max(wcc_sizes_erdos) + 2), align='left', rwidth=0.8, label='Erdos-Renyi graph')
plt.xlabel('WCC Size')
plt.ylabel('Count')
plt.title('WCC size Distribution of Erdos-Renyi graph')
plt.legend()
plt.show()

plt.hist(wcc_sizes_config, bins=range(1, max(wcc_sizes_config) + 2), align='left', rwidth=0.8, label='Configuration Model Graph')
plt.xlabel('WCC Size')
plt.ylabel('Count')
plt.title('WCC size distribution of configuration model graph')
plt.legend()
plt.show()








# Ex 31 "For each of these distributions, state whether or not the random models have the
# same property as the real world graph."

Real_G = nx.read_edgelist ('ca-AstroPh.txt')

# Generate Erdős-Rényi random graph
n = 18772  
m = 198110  

Erdos_G = nx.gnm_random_graph (n, m)

# Generate Configuration model random graph
degree_seq = [Real_G.degree (node) for node in Real_G.nodes ()]
Config_g = nx.configuration_model (degree_seq)

# Calculate degree distributions
degree_dist_real = nx.degree_histogram (Real_G)
degree_dist_erdos = nx.degree_histogram (Erdos_G)
degree_dist_config = nx.degree_histogram (Config_g)

# Calculate shortest path length distributions
shortest_paths_real = nx.shortest_path_length (Real_G)
path_lengths_real = [length for lengths in shortest_paths_real.values () for length in lengths.values ()]
shortest_paths_erdos = nx.shortest_path_length (Erdos_G)
path_lengths_erdos = [length for lengths in shortest_paths_erdos.values () for length in lengths.values ()]
shortest_paths_config = nx.shortest_path_length (Config_g)
path_lengths_config = [length for lengths in shortest_paths_config.values () for length in lengths.values ()]

# Calculate clustering coefficient distributions
clustering_coeffs_real = nx.clustering (Real_G)
clustering_coeffs_erdos = nx.clustering (Erdos_G)
clustering_coeffs_config = nx.clustering (Config_g)

# Calculate WCC size distributions
wcc_real = nx.weakly_connected_components (Real_G)
wcc_sizes_real = [len (wcc) for wcc in wcc_real]
wcc_erdos = nx.weakly_connected_components (Erdos_G)
wcc_sizes_erdos = [len (wcc) for wcc in wcc_erdos]
wcc_config = nx.weakly_connected_components (Config_g)
wcc_sizes_config = [len (wcc) for wcc in wcc_config]

# Writing the results
with open ('graph_properties.txt', 'w') as f:
    f.write (f'Degree distribution of real world graph: {np.array (degree_dist_real)}\n')
    f.write (f'Degree distribution of Erdos-Renyi graph: {np.array (degree_dist_erdos)}\n')
    f.write (f'Degree distribution of configuration model graph: {np.array (degree_dist_config)}\n\n')
    f.write (f'Shortest Path Length Distribution of real world graph: {np.array (path_lengths_real)}\n')
    f.write (f'Shortest Path Length Distribution of Erdos-Renyi graph: {np.array (path_lengths_erdos)}\n')
    f.write (f'Shortest Path Length Distribution of configuration model graph: {np.array (path_lengths_config)}\n\n')
    






# Ex 32 "Are the random graph generators capable of generating graphs that are representative of real world graphs?"

# URL of the dataset
url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
file_name = "ca-GrQc.txt.gz"

urllib.request.urlretrieve(url, file_name)

# Extract the gz file
with gzip.open(file_name, 'rb') as gz_file:
    with open("ca-GrQc.txt", 'wb') as output_file:
        output_file.write(gz_file.read())


Real_G = nx.read_edgelist("ca-GrQc.txt")

# Generate a small-world graph
G_S_W = nx.watts_strogatz_graph(len(Real_G.nodes()), len(Real_G.edges()), 0.1)

# Calculate the degree distributions
degree_dist_real = nx.degree_histogram(Real_G)
degree_dist_small_world = nx.degree_histogram(G_S_W)

# Plot the degree distributions
plt.figure(figsize=(10, 5))
plt.plot(degree_dist_real, label="The real world graph")
plt.plot(degree_dist_small_world, label="The small world graph")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree Distribution")
plt.legend()
plt.show()

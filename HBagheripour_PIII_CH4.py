# Social Network Chapter 4
# Haydeh Bagheripour
# Ex 33-38


import urllib.request
import matplotlib.pyplot as plt
import networkx as nx
import g

#Download the General Relativity and Quantum Cosmology collaboration network
#available at https://snap.stanford.edu/data/ca-GrQc.txt.gz.
#For the graph corresponding to this dataset (which will be referred to as real world
#graph), generate a small world graph and compute the following network parameters:

# Chapter 4 Problems:
# EX 33 "Degree distribution"

url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
file_name = "ca-GrQc.txt.gz"

# Download the above dataset
urllib.request.urlretrieve(url, file_name)

# Extract the gzipped file
with gzip.open(file_name, 'rb') as gz_file:
    with open("ca-GrQc.txt", 'wb') as output_file:
        output_file.write(gz_file.read())

Real_G = nx.read_edgelist("ca-GrQc.txt")

# Generate a small world graph
G_S_W = nx.watts_strogatz_graph(len(Real_G.nodes()), len(Real_G.edges()), 0.1)

# Calculate the degree distributions
degree_dist_real = nx.degree_histogram(Real_G)
degree_dist_small_world = nx.degree_histogram(G_S_W)

# Plot the degree distributions
plt.figure(figsize=(10, 5))
plt.plot(degree_dist_real, label=" The real world graph")
plt.plot(degree_dist_small_world, label="The small world graph")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree Distribution")
plt.legend()
plt.show()



# Ex 34 "Short path length distribution"


Real_G = nx.read_edgelist("ca-GrQc.txt")

# Compute the shortest path lengths
shortest_path_lengths = nx.shortest_path_length(Real_G)

# Count the occurrences of each path length
path_length_counts = {}
for source, paths in shortest_path_lengths:
    for target, length in paths.items():
        if length not in path_length_counts:
            path_length_counts[length] = 0
        path_length_counts[length] += 1

# Convert the counts to a distribution
total_paths = len(Real_G.nodes()) * (len(Real_G.nodes()) - 1)
path_length_dist = [path_length_counts[length] / total_paths for length in range(1, max(path_length_counts.keys()) + 1)]

# Plot the shortest path length distribution
plt.figure(figsize=(10, 5))
plt.plot(range(1, max(path_length_counts.keys()) + 1), path_length_dist)
plt.xlabel("The shortest path length")
plt.ylabel("Distribution")
plt.title("The shortest path length distribution")
plt.show()



# Ex 35 "Clustering coefficient distribution"

Real_G = nx.read_edgelist("ca-GrQc.txt")

# Compute the shortest path lengths
shortest_path_lengths = nx.shortest_path_length(Real_G)

# Count the occurrences of each path length
path_length_counts = {}
for source, paths in shortest_path_lengths:
    for target, length in paths.items():
        if length not in path_length_counts:
            path_length_counts[length] = 0
        path_length_counts[length] += 1

# Convert the counts to a distribution
total_paths = len(Real_G.nodes()) * (len(Real_G.nodes()) - 1)
path_length_dist = [path_length_counts[length] / total_paths for length in range(1, max(path_length_counts.keys()) + 1)]

# Plot the shortest path length distribution
plt.figure(figsize=(10, 5))
plt.plot(range(1, max(path_length_counts.keys()) + 1), path_length_dist)
plt.xlabel("The shortest path length")
plt.ylabel("Distribution")
plt.title("The shortest path length distribution")
plt.show()



# EX 36 "WCC size distribution"

Real_G = nx.read_edgelist("ca-GrQc.txt")

# Get the weakly connected components
wcc = nx.weakly_connected_components(Real_G)

# Compute the size of each weakly connected component
wcc_sizes = [len(component) for component in wcc]

# Count the occurrences of each component size
size_counts = {}
for size in wcc_sizes:
    if size not in size_counts:
        size_counts[size] = 0
    size_counts[size] += 1

# Convert the counts to a distribution
total_components = len(wcc_sizes)
size_dist = [size_counts[size] / total_components for size in sorted(size_counts.keys())]

# Plot 
plt.figure(figsize=(10, 5))
plt.plot(sorted(size_counts.keys()), size_dist)
plt.xlabel("Component Size")
plt.ylabel("Distribution")
plt.title("WCC Size Distribution")
plt.show()


# Ex 37 "For each of these distributions, state whether or not the small world model has
#the same property as the real world grap"
# Ex 38 "Is the small world graph generator capable of generating graphs that are representative of real world graphs?"


Real_G = nx.read_edgelist("ca-GrQc.txt")

# Get the weakly connected components
wcc = nx.weakly_connected_components(Real_G)

# Compute the size of each weakly connected component
wcc_sizes = [len(component) for component in wcc]

# Count the occurrences of each component size
size_counts = {}
for size in wcc_sizes:
    if size not in size_counts:
        size_counts[size] = 0
    size_counts[size] += 1

# Convert the counts to a distribution
total_components = len(wcc_sizes)
size_dist = [size_counts[size] / total_components for size in sorted(size_counts.keys())]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(sorted(size_counts.keys()), size_dist)
plt.xlabel("Component Size")
plt.ylabel("Distribution")
plt.title("WCC Size Distribution")
plt.show()


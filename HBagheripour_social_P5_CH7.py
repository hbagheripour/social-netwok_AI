# Social Network Chapter 7
# Haydeh Bagheripour
# Ex 47-49

# Problems:
# Download the signed Epinions social network dataset available at https://snap.
# stanford.edu/data/soc-sign-epinions.txt.gz.
# Consider the graph as undirected and compute the following:

import networkx as nx
import gzip
import random

# Ex 47:  "Calculate the count and fraction of triads of each type in this network."

G = nx.read_edgelist('soc-sign-epinions.txt', delimiter='\t', data=[('sign', int)], create_using=nx.Graph())

triangles = sum(nx.triangles(G).values())
total_triads = triangles // 3


for triad in range(1, 17):
    print(f"Triad {triad}:/{total_triads}")


    
'''result:Triad 1: count = 1, fraction = 1/4910076
Triad 2: count = 1, fraction = 1/4910076
Triad 3: count = 1, fraction = 1/4910076
Triad 4: count = 1, fraction = 1/4910076
Triad 5: count = 1, fraction = 1/4910076
Triad 6: count = 1, fraction = 1/4910076
Triad 7: count = 1, fraction = 1/4910076
Triad 8: count = 1, fraction = 1/4910076
Triad 9: count = 1, fraction = 1/4910076
Triad 10: count = 1, fraction = 1/4910076
Triad 11: count = 1, fraction = 1/4910076
Triad 12: count = 1, fraction = 1/4910076
Triad 13: count = 1, fraction = 1/4910076
Triad 14: count = 1, fraction = 1/4910076
Triad 15: count = 1, fraction = 1/4910076
Triad 16: count = 1, fraction = 1/4910076 '''








# Ex 48:  "Calculate the fraction of positive and negative edges in the graph. Let the fraction
# of positive edges be p. Assuming that each edge of a triad will independently be assigned a positive sign with probability p and a negative sign with probability
# 1 − p, calculate the probability of each type of triad."

num_of_edges = G.number_of_edges()
num_of_positive_edges = sum([1 for (_, _, sign) in G.edges(data='sign') if sign > 0])
num_of_negative_edges = sum([1 for (_, _, sign) in G.edges(data='sign') if sign < 0])
frac_positive_edges = num_of_positive_edges / num_of_edges
frac_negative_edges = num_of_negative_edges / num_of_edges

print(f"positive edges: {frac_positive_edges}")
print(f"negative edges: {frac_negative_edges}")


'''result:
Fraction of positive edges: 0.8324882724088661
Fraction of negative edges: 0.1675117275911338'''

p = frac_positive_edges
q = 1 - p

triad_probs = {}
triad_probs['003'] = p ** 3  
triad_probs['030T'] = q ** 3  
triad_probs['012'] = 3 * p ** 2 * q  
triad_probs['021U'] = p * q ** 2 + q * p ** 2  
triad_probs['102'] = 3 * p * q ** 2  
triad_probs['021D'] = p * q ** 2 + q * p ** 2  

for k, v in triad_probs.items():
    print(f"Triad {k}: probability = {v}")
    

'''result:
Triad 003: probability = 0.5769449448274973
Triad 012: probability = 0.3482753366124039
Triad 102: probability = 0.07007930951929901
Triad 021D: probability = 0.1394515487105676
Triad 021U: probability = 0.1394515487105676
Triad 030T: probability = 0.004700409040799899'''






# Ex 49:  "Now, analyse a simple generative model of signed networks by running simulations of the dynamic process on small networks in the following manner. Create a
# complete network on 10 nodes. For each edge, choose a sign with uniform probability. Run this dynamic process for a million iterations. Repeat this process 100 times.
# What fraction of these networks are balanced?"


n = 10
num_iterations = 1000000


def generate_random_signed_graph():
    G = nx.complete_graph(n)
    for u, v in G.edges():
        sign = random.choice([-1, 1])
        G[u][v]['sign'] = sign
    return G


def run_dynamic_process(G):
    states = {n: random.choice([-1, 1]) for n in G.nodes()}
    for u, v in G.edges():
        sign = G[u][v]['sign']
        if sign == 1:
            if states[u] == states[v]:
                states[u] *= -1
            else:
                pass
        else:
            if states[u] == states[v]:
                pass
            else:
                states[v] *= -1
    return all(s == 1 for s in states.values())


num_balanced = 0
for i in range(num_iterations):
    G = generate_random_signed_graph()
    if run_dynamic_process(G):
        num_balanced += 1

frac_balanced = num_balanced / num_iterations
print(f"balance networks: {frac_balanced:.4f}")



'''result:
Fraction of balanced networks: 0.0010'''



num_balanced = 0
num_iterations = 100

for i in range(num_iterations):
    G = generate_random_signed_graph()
    if run_dynamic_process(G):
        num_balanced += 1

frac_balanced = num_balanced / num_iterations
print(f"networks: {frac_balanced:.20f}")



'''result:
Fraction of balanced networks: 0.01000000000000000021'''
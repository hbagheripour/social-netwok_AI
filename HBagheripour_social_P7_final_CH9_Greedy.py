# Social Network 
# Haydeh Bagheripour

# Chapter9 "Influence Maximisation"
# Greedy Algorithm
# Final Project


import random

def influence_maximization_Greedy(nodes, edges, weights, k):
    
    """
    یافتن مجموعه بهترین گره‌ها برای حداکثر کردن تاثیر در یک گراف دارای وزن‌

    پارامتر‌ها:
        nodes (list): لیست گره‌ها در گراف.
        edges (list): لیست یال‌ها .
        weights (dict): وزن‌های هر گره .
        k (int): تعداد گره‌هایی که باید در مجموعه بهترین گره‌ها قرار گیرند.
    """

    seed_set = set()

    while len(seed_set) < k:
        max_node = None
        max_influence = -1

        for node in nodes:
            if node not in seed_set:
                influence = sc_influence(node, seed_set, nodes, edges, weights)
                if influence > max_influence:
                    max_influence = influence
                    max_node = node

        seed_set.add(max_node)

    return seed_set

def sc_influence(node, seed_set, nodes, edges, weights):
    activated_nodes = set()
    activated_nodes.add(node)
    new_activated = set()

    while new_activated:
        for activated_node in new_activated:
            for neighbor in edges[activated_node]:
                if neighbor not in activated_nodes:
                    threshold = random.uniform(0, 1)
                    weight_sum = sum(weights[neighbor][activated_neighbor] for activated_neighbor in activated_nodes)

                    if weight_sum >= threshold:
                        activated_nodes.add(neighbor)

        new_activated = activated_nodes - new_activated

    return len(activated_nodes)

# Example usage
nodes = ['A', 'B', 'C', 'D']
edges = {'A': ['B', 'C'], 'B': ['C', 'D'], 'C': ['D'], 'D': []}
weights = {'A': {'B': 0.6, 'C': 0.8}, 'B': {'C': 0.4, 'D': 0.5}, 'C': {'D': 0.7}, 'D': {}}
k = 2

seed_set = influence_maximization_Greedy(nodes, edges, weights, k)
print(seed_set)


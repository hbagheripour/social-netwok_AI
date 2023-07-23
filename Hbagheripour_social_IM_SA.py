# Social Network 
# Haydeh Bagheripour

# Chapter9 "Influence Maximisation"
# Simulated Annealing Algorithm
# Final Project

import random
import math

def get_influence(graph, seeds):
    visited = set(seeds)
    queue = list(seeds)
    influence = len(seeds)

    while queue:
        node = queue.pop(0)
        neighbors = graph.get(node, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                if random.random() <= graph[node][neighbor]:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    influence += 1

    return influence

def simulated_annealing(nodes, edges, weights, k, initial_temperature=100, cooling_rate=0.95, iterations=1000):
    def get_random_solution():
        return set(random.sample(nodes, k))

    def evaluate_solution(solution):
        return get_influence(graph, solution)

    def acceptance_probability(old_influence, new_influence, temperature):
        if new_influence > old_influence:
            return 1.0
        return math.exp((new_influence - old_influence) / temperature)

    graph = {}
    for i, (src, dst) in enumerate(edges):
        weight = weights[i]
        if src not in graph:
            graph[src] = {}
        if dst not in graph:
            graph[dst] = {}

        graph[src][dst] = weight
        graph[dst][src] = weight

    current_solution = get_random_solution()
    best_solution = current_solution.copy()
    current_influence = evaluate_solution(current_solution)
    best_influence = current_influence

    current_temperature = initial_temperature

    for _ in range(iterations):
        new_solution = get_random_solution()
        new_influence = evaluate_solution(new_solution)

        if acceptance_probability(current_influence, new_influence, current_temperature) > random.random():
            current_solution = new_solution
            current_influence = new_influence

        if current_influence > best_influence:
            best_solution = current_solution.copy()
            best_influence = current_influence

        current_temperature *= cooling_rate

    return best_solution

# نمونه ورودی‌ها
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'J'),
         ('A', 'E'), ('B', 'F'), ('C', 'G'), ('D', 'H'), ('E', 'I')]
weights = [0.5, 0.3, 0.8, 0.4, 0.6, 0.2, 0.7, 0.9, 0.3, 0.6, 0.4, 0.5, 0.2, 0.7, 0.8]
k = 3

result = simulated_annealing(nodes, edges, weights, k)
print("بهترین رئوس جهت تأثیرگذاری: ", result)

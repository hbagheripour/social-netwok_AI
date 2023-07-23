# Social Network 
# Haydeh Bagheripour

# Chapter9 "Influence Maximisation" 
# linear_threshold_model
# Final Project 

import random

def linear_threshold_model(graph, thresholds, weights, seed_set):
    """
    محاسبه مدل تاثیرگذاری انتشار با مدل آستانه خطی بر روی یک گراف.

    پارامتر‌ها:
        graph (dict): یک گراف به صورت دیکشنری که گره‌ها به عنوان کلیدها و لیست مجاورت به عنوان مقدارها ذخیره می‌شوند.
        thresholds (dict): آستانه‌های هر گره .
        weights (dict): وزن‌های هر گره .
        seed_set (list): مجموعه گره‌های اولیه که به عنوان نقطه شروع برای انتشار تاثیر استفاده می‌شوند.

    خروجی:
        activated_nodes (set): مجموعه گره‌هایی که در نتیجه انتشار تاثیر، فعال شده‌اند.
    """
    activated_nodes = set(seed_set)  # مجموعه‌ی گره‌هایی که تا الان فعال شده‌اند.
    new_activated_nodes = set(seed_set)  # مجموعه‌ی گره‌های جدیدی که در هر مرحله فعال می‌شوند.

    while new_activated_nodes:
        current_activated_nodes = new_activated_nodes.copy()
        new_activated_nodes.clear()

        # بررسی همسایه‌های گره‌های فعال در مرحله فعلی و احتمال فعال شدن آن‌ها.
        for node in current_activated_nodes:
            neighbors = graph.get(node, [])  # همسایه‌های گره فعلی.
            for neighbor in neighbors:
                if neighbor in activated_nodes:
                    continue  # گره‌هایی که قبلاً فعال شده‌اند را نادیده بگیریم.

                # محاسبه مقدار تاثیر با توجه به آستانه و وزن گره.
                activated_neighbors = [n for n in neighbors if n in activated_nodes]
                total_weight = sum(weights.get(n, 1) for n in activated_neighbors)
                if total_weight >= thresholds.get(neighbor, 0):
                    new_activated_nodes.add(neighbor)

        # افزودن گره‌های جدید به مجموعه‌ی کلی گره‌های فعال.
        activated_nodes.update(new_activated_nodes)

    return activated_nodes

#نمونه ورودی ها :
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['E', 'F'],
        'C': ['G', 'H'],
        'D': ['I', 'J'],
        'E': ['K', 'L'],
        'F': ['M', 'N'],
        'G': ['O', 'P'],
        'H': ['Q', 'R'],
        'I': ['S', 'T'],
        'J': ['U', 'V'],
        'K': [],
        'L': [],
        'M': [],
        'N': [],
        'O': [],
        'P': [],
        'Q': [],
        'R': [],
        'S': [],
        'T': [],
        'U': [],
        'V': []
    }
    thresholds = {
        'A': 0.6,
        'B': 0.5,
        'C': 0.7,
        'D': 0.4,
        'E': 0.8,
        'F': 0.3,
        'G': 0.5,
        'H': 0.6,
        'I': 0.7,
        'J': 0.4,
        'K': 0.2,
        'L': 0.1,
        'M': 0.3,
        'N': 0.5,
        'O': 0.6,
        'P': 0.4,
        'Q': 0.2,
        'R': 0.1,
        'S': 0.3,
        'T': 0.5,
        'U': 0.6,
        'V': 0.4,
    }
    weights = {
        'A': 0.8,
        'B': 0.6,
        'C': 0.7,
        'D': 0.5,
        'E': 0.3,
        'F': 0.4,
        'G': 0.6,
        'H': 0.5,
        'I': 0.7,
        'J': 0.4,
        'K': 0.8,
        'L': 0.6,
        'M': 0.7,
        'N': 0.5,
        'O': 0.3,
        'P': 0.4,
        'Q': 0.6,
        'R': 0.5,
        'S': 0.7,
        'T': 0.4,
        'U': 0.8,
        'V': 0.6
    }
    seed_set = ['A', 'F', 'K']

    activated_nodes = linear_threshold_model(graph, thresholds, weights, seed_set)
    print("گره‌های فعال شده:", activated_nodes)
    
    
    '''Result:
    گره‌های فعال شده: {'F', 'A', 'L', 'B', 'E', 'M', 'K'} '''
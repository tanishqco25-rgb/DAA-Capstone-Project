"""
Mini-Courier Planner
Course: Design & Analysis of Algorithms (ETCCDA302)
A modular demonstration of sorting, 0/1 knapsack, Dijkstra and
Nearest-Neighbour TSP approximation.

Run:
    python mini_courier_planner.py
    python mini_courier_planner.py --benchmark

No third-party packages are required.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
import json
import random
import time


@dataclass
class Parcel:
    parcel_id: str
    destination: str
    weight: int
    priority: int
    deadline_hour: int
    value: int


def merge_sort_parcels(parcels: List[Parcel]) -> List[Parcel]:
    """Sort by priority (descending), then earlier deadline (ascending)."""
    if len(parcels) <= 1:
        return parcels[:]
    mid = len(parcels) // 2
    left = merge_sort_parcels(parcels[:mid])
    right = merge_sort_parcels(parcels[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a = (-left[i].priority, left[i].deadline_hour, left[i].parcel_id)
        b = (-right[j].priority, right[j].deadline_hour, right[j].parcel_id)
        if a <= b:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]


def knapsack_01(parcels: List[Parcel], capacity: int):
    """Return maximum-value parcel subset using 0/1 dynamic programming."""
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        p = parcels[i - 1]
        for cap in range(capacity + 1):
            dp[i][cap] = dp[i - 1][cap]
            if p.weight <= cap:
                dp[i][cap] = max(
                    dp[i][cap],
                    p.value + dp[i - 1][cap - p.weight]
                )
    chosen = []
    cap = capacity
    for i in range(n, 0, -1):
        if dp[i][cap] != dp[i - 1][cap]:
            p = parcels[i - 1]
            chosen.append(p)
            cap -= p.weight
    chosen.reverse()
    return chosen, dp[n][capacity], dp


def dijkstra(graph: Dict[str, Dict[str, int]], source: str):
    """Shortest distances and predecessors for a non-negative weighted graph."""
    unvisited = set(graph)
    distance = {node: float("inf") for node in graph}
    previous = {node: None for node in graph}
    distance[source] = 0
    while unvisited:
        current = min(unvisited, key=lambda node: distance[node])
        if distance[current] == float("inf"):
            break
        unvisited.remove(current)
        for neighbour, weight in graph[current].items():
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative edge weights.")
            candidate = distance[current] + weight
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = current
    return distance, previous


def reconstruct_path(previous, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = previous[node]
    return list(reversed(path))


def nearest_neighbour_tour(graph, start):
    """Greedy TSP approximation using shortest-path distances between stops."""
    nodes = list(graph.keys())
    remaining = set(nodes) - {start}
    tour, total = [start], 0
    current = start
    leg_details = []
    while remaining:
        distances, previous = dijkstra(graph, current)
        reachable = [(distances[node], node) for node in remaining
                     if distances[node] != float("inf")]
        if not reachable:
            raise ValueError("Not all delivery points are reachable from the current stop.")
        leg_distance, next_node = min(reachable, key=lambda item: (item[0], item[1]))
        path = reconstruct_path(previous, next_node)
        leg_details.append((current, next_node, leg_distance, path))
        total += leg_distance
        tour.append(next_node)
        remaining.remove(next_node)
        current = next_node
    distances, previous = dijkstra(graph, current)
    if distances[start] == float("inf"):
        raise ValueError("The route cannot return to the depot.")
    return_distance = distances[start]
    return_path = reconstruct_path(previous, start)
    leg_details.append((current, start, return_distance, return_path))
    total += return_distance
    tour.append(start)
    return tour, total, leg_details


def load_sample_data():
    parcels = [
        Parcel("P101", "A", 4, 5, 10, 80),
        Parcel("P102", "B", 6, 4, 12, 90),
        Parcel("P103", "C", 3, 5, 9, 70),
        Parcel("P104", "D", 5, 3, 15, 60),
        Parcel("P105", "E", 2, 2, 11, 35),
        Parcel("P106", "B", 4, 4, 10, 65),
    ]
    graph = {
        "Depot": {"A": 4, "B": 7, "C": 9, "D": 10, "E": 8},
        "A": {"Depot": 4, "B": 3, "C": 5, "D": 8, "E": 7},
        "B": {"Depot": 7, "A": 3, "C": 4, "D": 6, "E": 5},
        "C": {"Depot": 9, "A": 5, "B": 4, "D": 3, "E": 6},
        "D": {"Depot": 10, "A": 8, "B": 6, "C": 3, "E": 4},
        "E": {"Depot": 8, "A": 7, "B": 5, "C": 6, "D": 4},
    }
    return parcels, graph


def run_demo():
    parcels, graph = load_sample_data()
    capacity = 12
    print("\n=== MINI-COURIER PLANNER ===")
    print(f"Vehicle parcel capacity: {capacity} kg")
    sorted_parcels = merge_sort_parcels(parcels)
    print("\n1. Prioritized parcels (priority desc, deadline asc):")
    for p in sorted_parcels:
        print(f"  {p.parcel_id}: priority={p.priority}, deadline={p.deadline_hour}:00, "
              f"weight={p.weight}kg, value={p.value}")
    selected, total_value, dp = knapsack_01(sorted_parcels, capacity)
    print("\n2. Knapsack packing plan:")
    print("  Selected:", ", ".join(p.parcel_id for p in selected))
    print("  Total weight:", sum(p.weight for p in selected), "kg")
    print("  Total value:", total_value)
    distances, previous = dijkstra(graph, "Depot")
    print("\n3. Dijkstra shortest distances from Depot:")
    for node in graph:
        print(f"  Depot -> {node}: {distances[node]}")
    tour, route_length, legs = nearest_neighbour_tour(graph, "Depot")
    print("\n4. Nearest-Neighbour delivery tour:")
    print("  " + " -> ".join(tour))
    print("  Total route length:", route_length, "distance units")
    print("\n5. Leg details:")
    for start, end, dist, path in legs:
        print(f"  {start} -> {end}: {dist} via {' -> '.join(path)}")
    return parcels, graph, sorted_parcels, selected, total_value, tour, route_length


def benchmark():
    """Repeat algorithm timings; values are measured on the executing machine."""
    random.seed(42)
    sizes = [20, 40, 80, 120, 160]
    rows = []
    for n in sizes:
        parcels = [Parcel(f"P{i:04d}", f"D{i%12}", random.randint(1, 10),
                          random.randint(1, 5), random.randint(8, 20),
                          random.randint(10, 100)) for i in range(n)]
        t0 = time.perf_counter()
        merge_sort_parcels(parcels)
        sort_ms = (time.perf_counter() - t0) * 1000

        capacity = 100
        t0 = time.perf_counter()
        knapsack_01(parcels, capacity)
        knapsack_ms = (time.perf_counter() - t0) * 1000

        # Dense undirected graph with n vertices; integer non-negative weights.
        names = [f"N{i}" for i in range(n)]
        graph = {name: {} for name in names}
        for i in range(n):
            for j in range(i + 1, n):
                w = random.randint(1, 30)
                graph[names[i]][names[j]] = w
                graph[names[j]][names[i]] = w
        t0 = time.perf_counter()
        dijkstra(graph, names[0])
        dijkstra_ms = (time.perf_counter() - t0) * 1000
        rows.append((n, sort_ms, knapsack_ms, dijkstra_ms))
    print("\n=== BENCHMARK (milliseconds; measured locally) ===")
    print("n, merge_sort_ms, knapsack_ms(capacity=100), dijkstra_dense_ms")
    for row in rows:
        print(f"{row[0]}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-Courier Planner")
    parser.add_argument("--benchmark", action="store_true",
                        help="run algorithm timing tests")
    args = parser.parse_args()
    if args.benchmark:
        benchmark()
    else:
        run_demo()

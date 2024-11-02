import sys
import time
from collections import defaultdict

# Read and parse the .uwg format from standard input
def read_graph_from_input():
    input_data = sys.stdin.read().strip().splitlines()
    n = int(input_data[0].strip())    # Number of vertices
    m = int(input_data[1].strip())    # Number of edges
    edges = []
    for i in range(m):
        u, v, w = map(int, input_data[i + 2].strip().split())
        edges.append((u, v, w, i + 1))   # Add edge index for output convenience
    return n, m, edges

# Helper function to find the MST weight as an initial bound
def compute_mst_weight(n, edges):
    parent = list(range(n + 1))  # Union-Find data structure
    rank = [0] * (n + 1)
    mst_weight = 0
    sorted_edges = sorted(edges, key=lambda x: x[2])  # Sort by weight

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    # Kruskal's algorithm for MST
    for u, v, w, _ in sorted_edges:
        if find(u) != find(v):
            union(u, v)
            mst_weight += w
    return mst_weight

# Decision function A_d to determine if a spanning tree exists under a bound B
def decision_algorithm(n, edges, B):
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    weight_sum = 0
    edge_count = 0

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
            return True
        return False

    for u, v, w, _ in sorted(edges, key=lambda x: x[2]):
        if w > B:
            break
        if union(u, v):
            weight_sum += w
            edge_count += 1
            if weight_sum > B:
                return False

    return edge_count == n - 1

# Heuristic to assign scores to edges for guiding tree construction
def heuristic_value(edge, edges, degrees):
    u, v, weight, _ = edge
    mirror_weight = find_mirror_weight(u, v, edges)
    degree_sum = degrees[u] + degrees[v]
    alpha, beta = 1, 1
    return alpha * degree_sum + beta * abs(weight - mirror_weight)

def find_mirror_weight(u, v, edges):
    for x, y, w, _ in edges:
        if (x, y) == (v, u) or (y, x) == (u, v):
            return w
    return float('inf')

# Backtracking with heuristics to construct the spanning tree under B
def construct_tree_with_b(n, edges, B):
    degrees = defaultdict(int)
    for u, v, _, _ in edges:
        degrees[u] += 1
        degrees[v] += 1

    # Sort edges by weight first, then apply heuristic
    edges_sorted = sorted(edges, key=lambda e: (e[2], heuristic_value(e, edges, degrees)))
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    result_tree = []
    total_weight = 0

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
            return True
        return False

    for u, v, w, index in edges_sorted:
        if union(u, v):
            result_tree.append(index)
            total_weight += w

            # Stop if a valid spanning tree of n-1 edges is formed
            if len(result_tree) == n - 1:
                # Check if the total weight of the tree is within B
                if total_weight <= B:
                    return result_tree, total_weight
                else:
                    return None, None

    return None, None

# Main algorithm implementing Tasks (f) and (h)
def mfmst_algorithm(n, m, edges):
    W = compute_mst_weight(n, edges)
    B_low, B_high = 0, W
    while B_low < B_high:
        B_mid = (B_low + B_high) // 2
        if decision_algorithm(n, edges, B_mid):
            B_high = B_mid
        else:
            B_low = B_mid + 1
    optimal_B = B_low
    result_tree, tree_weight = construct_tree_with_b(n, edges, optimal_B)
    if result_tree is None:
        print("NO")
    else:
        # Output each edge index in the result tree
        for edge in result_tree:
            print(edge)
        # Output the total weight of the spanning tree
        print(tree_weight)

if __name__ == "__main__":
    n, m, edges = read_graph_from_input()
    mfmst_algorithm(n, m, edges)

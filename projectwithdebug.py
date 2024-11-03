import sys
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
    print("Computed MST weight:", mst_weight, file=sys.stderr)  # Debugging output
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
                print(f"Decision algorithm exceeded B={B} with weight_sum={weight_sum}", file=sys.stderr)
                return False

    result = edge_count == n - 1
    print(f"Decision algorithm result for B={B}: {result} with weight_sum={weight_sum}", file=sys.stderr)
    return result

# Backtracking with heuristics to construct the spanning tree under B
def construct_tree_with_b(n, edges, B):
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

    edges_sorted = sorted(edges, key=lambda e: e[2])  # Sort edges by weight

    for u, v, w, index in edges_sorted:
        if union(u, v):
            result_tree.append(index)
            total_weight += w
            print(f"Construct tree: Added edge {index} ({u}-{v}) with weight {w}, total_weight={total_weight}", file=sys.stderr)

            if total_weight > B:  # Early exit if weight exceeds B
                print("Construction failed: Total weight exceeded B", file=sys.stderr)
                return None, None

            # Stop if a valid spanning tree of n-1 edges is formed
            if len(result_tree) == n - 1:
                if total_weight <= B:
                    return result_tree, total_weight
                else:
                    print("Final check: total weight exceeded B after forming tree", file=sys.stderr)
                    return None, None

    print("Construct tree failed: No valid tree found within B", file=sys.stderr)
    return None, None

# Main algorithm implementing Tasks (f) and (h)
def mfmst_algorithm(n, m, edges):
    W = compute_mst_weight(n, edges)
    B_low, B_high = 0, W
    while B_low < B_high:
        B_mid = (B_low + B_high) // 2
        print(f"Binary search: B_low={B_low}, B_mid={B_mid}, B_high={B_high}", file=sys.stderr)
        if decision_algorithm(n, edges, B_mid):
            B_high = B_mid
        else:
            B_low = B_mid + 1
    optimal_B = B_low
    print(f"Optimal B found: {optimal_B}", file=sys.stderr)
    result_tree, tree_weight = construct_tree_with_b(n, edges, optimal_B)
    if result_tree is None:
        print("NO")
    else:
        for edge in result_tree:
            print(edge)
        print(tree_weight)

if __name__ == "__main__":
    n, m, edges = read_graph_from_input()
    mfmst_algorithm(n, m, edges)

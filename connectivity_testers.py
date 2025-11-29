import time
import random
import math
from collections import deque
from Implementation.graphs_helpers import (
    get_cached_graph,
    clear_graph_cache,
    get_neighbors,
)

CONFIG = {
    "epsilon": 0.001,
    "constant_c_simple": 3,
    "constant_c_improved": 3,
}


def basic_BFS_connectivity_tester(name: str) -> bool:
    """
    Basic connectivity tester using full BFS traversal.
    This is O(n) and visits all reachable vertices from a random start vertex.

    Args:
        name: Name of the graph file (without extension)

    Returns:
        True if graph is connected, False otherwise
    """
    # Load the graph (cached helper in graphs_helpers)
    adjacency, n, d, c, component_sizes = get_cached_graph(name)

    # Format component sizes as string
    sizes_str = ", ".join(str(s) for s in component_sizes)

    print(f"=== Basic BFS Connectivity Tester ===")
    print(f"Graph: {name}")
    print(f"  n={n}, d={d}, c={c}")
    print(f"  Component sizes: {sizes_str}")

    # Start timing
    start_time = time.perf_counter()

    # Select a random starting vertex uniformly at random
    start_vertex = random.randint(0, n - 1)

    # BFS traversal
    visited = set()
    queue = deque([start_vertex])
    visited.add(start_vertex)

    while queue:
        v = queue.popleft()
        neighbors = get_neighbors(adjacency, v)
        for u in neighbors:
            if u not in visited:
                visited.add(u)
                queue.append(u)

    # End timing
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Check if all vertices were visited
    component_size = len(visited)

    print(f"  Start vertex: {start_vertex}")
    print(f"  Vertices visited: {component_size}")
    print(f"  Time elapsed: {elapsed_time:.6f} seconds")

    if component_size == n:
        print(f"  Result: CONNECTED")
        return True
    else:
        print(f"  Result: NOT CONNECTED")
        print(f"  Component size from start vertex: {component_size} (out of {n})")
        return False


def simple_connectivity_tester(name: str) -> bool:
    """
    Simple connectivity tester based on Algorithm 4.1 from the paper.
    Uses epsilon-far testing with bounded BFS iterations.

    The algorithm:
    - Performs t = c / (epsilon * d) iterations
    - In each iteration, samples a random vertex and runs BFS
    - BFS stops when reaching 2 / (epsilon * d) vertices or no new vertices
    - If a small component (< n vertices) is found, reject (return False)
    - If no small component found after all iterations, accept (return True)

    Args:
        name: Name of the graph file (without extension)

    Returns:
        True if graph appears connected (or epsilon-close), False if epsilon-far
    """
    # Load the graph (cached helper in graphs_helpers)
    adjacency, n, d, c, component_sizes = get_cached_graph(name)

    # Get parameters from CONFIG
    epsilon = CONFIG["epsilon"]
    constant_c = CONFIG["constant_c_simple"]

    # Calculate algorithm parameters
    # Threshold for small components: 2 / (epsilon * d)
    vertex_threshold = int(math.ceil(2 / (epsilon * d)))

    # Number of iterations: c / (epsilon * d) where c is the hidden constant
    num_iterations = int(math.ceil(constant_c / (epsilon * d)))

    # Calculate theoretical threshold for epsilon-farness (Theorem 3.1)
    # Graph is epsilon-far if it has MORE than epsilon * d * n components
    epsilon_far_threshold = epsilon * d * n
    is_epsilon_far = c > epsilon_far_threshold
    target_result = (
        "EPSILON-FAR (should reject)"
        if is_epsilon_far
        else "EPSILON-CLOSE (should accept)"
    )

    # Calculate the minimal epsilon that would classify this graph as epsilon-far
    # From c > epsilon * d * n, we get epsilon < c / (d * n)
    epsilon_for_far = c / (d * n)

    # Format component sizes as string
    sizes_str = ", ".join(str(s) for s in component_sizes)

    print(f"=== Simple Connectivity Tester (Algorithm 4.1) ===")
    print(f"Graph: {name}")
    print(f"  n={n}, d={d}, c={c}")
    print(f"  Component sizes: {sizes_str}")
    print(f"  epsilon={epsilon}, constant_c={constant_c}")
    print(f"  Vertex threshold (2/(ε·d)): {vertex_threshold}")
    print(f"  Number of iterations (c/(ε·d)): {num_iterations}")
    print(f"\n  --- Theoretical Analysis (Theorem 3.1) ---")
    print(f"  ε·d·n threshold: {epsilon_far_threshold:.2f}")
    print(f"  Actual components: {c}")
    print(f"  Target result: {target_result}")
    print(f"  ε needed for ε-far: < {epsilon_for_far:.2e} (current ε={epsilon})")
    print(f"  ---------------------------------------------\n")

    # Start timing
    start_time = time.perf_counter()

    small_component_found = False
    total_vertices_visited = 0

    for iteration in range(num_iterations):
        # Select a vertex uniformly at random
        start_vertex = random.randint(0, n - 1)

        # Bounded BFS traversal
        visited = set()
        queue = deque([start_vertex])
        visited.add(start_vertex)

        while queue and len(visited) < vertex_threshold:
            v = queue.popleft()
            neighbors = get_neighbors(adjacency, v)
            for u in neighbors:
                if u not in visited:
                    visited.add(u)
                    if len(visited) >= vertex_threshold:
                        break
                    queue.append(u)

        component_size = len(visited)
        total_vertices_visited += component_size

        # Check if we found a small connected component
        # A component is "small" if:
        # 1. BFS terminated before reaching threshold (no more vertices to visit)
        # 2. The component size is less than n (not the entire graph)
        if len(queue) == 0 and component_size < vertex_threshold and component_size < n:
            # Found a small connected component - graph is epsilon-far from connected
            small_component_found = True
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print(
                f"  Iteration {iteration + 1}: Found small component of size {component_size}"
            )
            print(
                f"  Total vertices visited across iterations: {total_vertices_visited}"
            )
            print(f"  Time elapsed: {elapsed_time:.6f} seconds")
            print(f"  Algorithm result: NOT CONNECTED (epsilon-far)")
            print(
                f"  Match with target: {'YES' if is_epsilon_far else 'NO (impossible - one-sided error)'}"
            )
            return False

    # End timing
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"  Total vertices visited across iterations: {total_vertices_visited}")
    print(f"  Time elapsed: {elapsed_time:.6f} seconds")
    print(f"  Algorithm result: CONNECTED (or epsilon-close)")
    print(
        f"  Match with target: {'YES' if not is_epsilon_far else 'NO (false negative - missed small component)'}"
    )
    return True


def improved_connectivity_tester(name: str) -> bool:
    """
    Improved connectivity tester based on Algorithm 4.2 from the paper.
    Uses logarithmic component size intervals for better efficiency.

    The algorithm:
    - Computes ℓ = ⌈log₂(2/(ε·d) + 1)⌉ intervals for component sizes
    - For each interval i ∈ [1, ℓ]:
      - Performs tᵢ = c·ℓ/(2ⁱ·ε·d) iterations
      - BFS stops when reaching 2ⁱ vertices or no new vertices
    - Components in Cᵢ have between 2^(i-1) and 2^i - 1 vertices
    - If a small component (< n) is found, reject (return False)
    - If no small component found, accept (return True)

    Args:
        name: Name of the graph file (without extension)

    Returns:
        True if graph appears connected (or epsilon-close), False if epsilon-far
    """
    # Load the graph (cached helper in graphs_helpers)
    adjacency, n, d, c, component_sizes = get_cached_graph(name)

    # Get parameters from CONFIG
    epsilon = CONFIG["epsilon"]
    constant_c = CONFIG["constant_c_improved"]

    # Calculate ℓ: number of component size intervals
    # ℓ = ⌈log₂(2/(ε·d) + 1)⌉
    ell = int(math.ceil(math.log2(2 / (epsilon * d) + 1)))

    # Calculate theoretical threshold for epsilon-farness (Theorem 3.1)
    # Graph is epsilon-far if it has MORE than epsilon * d * n components
    epsilon_far_threshold = epsilon * d * n
    is_epsilon_far = c > epsilon_far_threshold
    target_result = (
        "EPSILON-FAR (should reject)"
        if is_epsilon_far
        else "EPSILON-CLOSE (should accept)"
    )

    # Calculate the minimal epsilon that would classify this graph as epsilon-far
    # From c > epsilon * d * n, we get epsilon < c / (d * n)
    epsilon_for_far = c / (d * n)

    # Format component sizes as string
    sizes_str = ", ".join(str(s) for s in component_sizes)

    print(f"=== Improved Connectivity Tester (Algorithm 4.2) ===")
    print(f"Graph: {name}")
    print(f"  n={n}, d={d}, c={c}")
    print(f"  Component sizes: {sizes_str}")
    print(f"  epsilon={epsilon}, constant_c={constant_c}")
    print(f"  ℓ (number of intervals): {ell}")

    # Print interval information
    print(f"  Intervals:")
    total_planned_iterations = 0
    for i in range(1, ell + 1):
        vertex_threshold_i = 2**i
        iterations_i = int(math.ceil((constant_c * ell) / ((2**i) * epsilon * d)))
        size_lower = 2 ** (i - 1)
        size_upper = (2**i) - 1
        total_planned_iterations += iterations_i
        print(
            f"    C_{i}: sizes [{size_lower}, {size_upper}], threshold={vertex_threshold_i}, iterations={iterations_i}"
        )

    print(f"  Total planned iterations: {total_planned_iterations}")
    print(f"\n  --- Theoretical Analysis (Theorem 3.1) ---")
    print(f"  ε·d·n threshold: {epsilon_far_threshold:.2f}")
    print(f"  Actual components: {c}")
    print(f"  Target result: {target_result}")
    print(f"  ε needed for ε-far: < {epsilon_for_far:.2e} (current ε={epsilon})")
    print(f"  ---------------------------------------------\n")

    # Start timing
    start_time = time.perf_counter()

    total_vertices_visited = 0
    total_iterations_done = 0

    # Outer loop: iterate over component size intervals
    for i in range(1, ell + 1):
        # Vertex threshold for this interval: 2^i
        vertex_threshold = 2**i

        # Number of iterations for this interval: c·ℓ/(2^i·ε·d)
        num_iterations = int(math.ceil((constant_c * ell) / ((2**i) * epsilon * d)))

        # Inner loop: sample vertices and run bounded BFS
        for iteration in range(num_iterations):
            total_iterations_done += 1

            # Select a vertex uniformly at random
            start_vertex = random.randint(0, n - 1)

            # Bounded BFS traversal
            visited = set()
            queue = deque([start_vertex])
            visited.add(start_vertex)

            while queue and len(visited) < vertex_threshold:
                v = queue.popleft()
                neighbors = get_neighbors(adjacency, v)
                for u in neighbors:
                    if u not in visited:
                        visited.add(u)
                        if len(visited) >= vertex_threshold:
                            break
                        queue.append(u)

            component_size = len(visited)
            total_vertices_visited += component_size

            # Check if we found a small connected component
            # A component is "small" if:
            # 1. BFS terminated before reaching threshold (queue empty)
            # 2. The component size is less than n (not the entire graph)
            if (
                len(queue) == 0
                and component_size < vertex_threshold
                and component_size < n
            ):
                # Found a small connected component - graph is epsilon-far from connected
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                print(
                    f"  Interval {i}, Iteration {iteration + 1}: Found small component of size {component_size}"
                )
                print(f"  Total iterations done: {total_iterations_done}")
                print(
                    f"  Total vertices visited across iterations: {total_vertices_visited}"
                )
                print(f"  Time elapsed: {elapsed_time:.6f} seconds")
                print(f"  Algorithm result: NOT CONNECTED (epsilon-far)")
                print(
                    f"  Match with target: {'YES' if is_epsilon_far else 'NO (impossible - one-sided error)'}"
                )
                return False

    # End timing
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"  Total iterations done: {total_iterations_done}")
    print(f"  Total vertices visited across iterations: {total_vertices_visited}")
    print(f"  Time elapsed: {elapsed_time:.6f} seconds")
    print(f"  Algorithm result: CONNECTED (or epsilon-close)")
    print(
        f"  Match with target: {'YES' if not is_epsilon_far else 'NO (false negative - missed small component)'}"
    )
    return True


if __name__ == "__main__":
    # Test with example graphs - Basic BFS Tester

    print("\n" + "=" * 50)
    basic_BFS_connectivity_tester("graph_2000000_20_20")

    print("\n" + "=" * 50)
    basic_BFS_connectivity_tester("graph_2000000_20_5")

    # Test with example graphs - Simple Tester

    print("\n" + "=" * 50)
    simple_connectivity_tester("graph_2000000_20_20")

    print("\n" + "=" * 50)
    simple_connectivity_tester("graph_2000000_20_5")

    # Test with example graphs - Improved Tester

    print("\n" + "=" * 50)
    improved_connectivity_tester("graph_2000000_20_20")

    print("\n" + "=" * 50)
    improved_connectivity_tester("graph_2000000_20_5")

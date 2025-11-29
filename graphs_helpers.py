from pathlib import Path
import numpy as np
from typing import Optional
import random
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

GRAPHS_PATH = Path(__file__).resolve().parent / "stored_graphs"
VISUALS_PATH = Path(__file__).resolve().parent / "stored_visuals"
GRAPHS_PATH.mkdir(parents=True, exist_ok=True)
VISUALS_PATH.mkdir(parents=True, exist_ok=True)

SEPARATE_COMPONENTS_IN_VISUALS = True
MAX_PARALLEL_WORKERS = 10

# Minimum vertices per component to justify multiprocessing overhead
MIN_VERTICES_FOR_PARALLEL = 10000

# In-memory cache for loaded graphs (shared helper)
_graph_cache = {}


def get_cached_graph(name: str) -> tuple:
    """
    Load a graph from cache or disk. Caches the result for future calls.

    Returns: (adjacency, n, d, c, component_sizes)
    """
    if name not in _graph_cache:
        _graph_cache[name] = load_graph(name)
    return _graph_cache[name]


def clear_graph_cache(name: str = None):
    """
    Clear the graph cache. If name is None clears all, otherwise clears single entry.
    """
    if name is None:
        _graph_cache.clear()
    else:
        _graph_cache.pop(name, None)


def create_and_store_graph(n: int, d: int, c: int, name: str) -> Path:
    """
    Create a bounded-degree graph and store it efficiently.

    Args:
        n: Total number of vertices
        d: Degree bound (max neighbors per vertex)
        c: Number of connected components
        name: Name for the graph file (without extension)

    Returns:
        Path to the stored graph file

    The graph is stored as .npz with:
        - adjacency: flattened adjacency list (n x d array, 0 = no neighbor)
        - metadata: array [n, d, c]
        - component_sizes: array of component sizes
    """
    if c > n:
        raise ValueError(f"Cannot have more components ({c}) than vertices ({n})")
    if d < 1:
        raise ValueError(f"Degree bound must be at least 1, got {d}")
    if c < 1:
        raise ValueError(f"Must have at least 1 component, got {c}")

    # Generate the graph as adjacency list representation
    adjacency, component_sizes = _generate_bounded_degree_graph(n, d, c)

    # Store the graph
    filepath = GRAPHS_PATH / f"{name}.npz"
    np.savez_compressed(
        filepath,
        adjacency=adjacency,
        metadata=np.array([n, d, c], dtype=np.int64),
        component_sizes=np.array(component_sizes, dtype=np.int64),
    )

    return filepath


def _generate_bounded_degree_graph(n: int, d: int, c: int) -> tuple:
    """
    Generate a bounded-degree graph with c connected components.
    Uses parallel processing for large graphs with multiple components.

    Returns:
        tuple: (adjacency, component_sizes) where:
            - adjacency: np.ndarray of shape (n, d), 1-indexed neighbors (0 = no edge)
            - component_sizes: list of component sizes
    """
    # Pre-calculate component sizes before any parallel work
    component_sizes = _distribute_vertices_to_components(n, c)

    # Create component vertex ranges (start_idx, size) for each component
    component_ranges = []
    component_start = 0
    for size in component_sizes:
        component_ranges.append((component_start, size))
        component_start += size

    # Decide whether to use parallel processing
    use_parallel = c > 1 and n >= MIN_VERTICES_FOR_PARALLEL and MAX_PARALLEL_WORKERS > 1

    # Initialize adjacency list
    adjacency = np.zeros((n, d), dtype=np.int32)

    if use_parallel:
        # Process components in parallel
        num_workers = min(MAX_PARALLEL_WORKERS, c)

        # Create worker arguments: (start_idx, size, d, random_seed)
        # Each worker gets a unique seed for reproducibility
        base_seed = random.randint(0, 2**31 - 1)
        worker_args = [
            (start_idx, size, d, base_seed + i)
            for i, (start_idx, size) in enumerate(component_ranges)
        ]

        with Pool(processes=num_workers) as pool:
            results = pool.map(_generate_component_edges, worker_args)

        # Merge results into main adjacency matrix
        for (start_idx, size), (local_adj, local_degrees) in zip(
            component_ranges, results
        ):
            adjacency[start_idx : start_idx + size, :] = local_adj
    else:
        # Sequential processing for small graphs
        degrees = np.zeros(n, dtype=np.int32)
        for start_idx, size in component_ranges:
            if size < 2:
                continue
            vertices = list(range(start_idx, start_idx + size))
            _connect_component(vertices, adjacency, degrees, d)

    return adjacency, component_sizes


def _generate_component_edges(args: tuple) -> tuple:
    """
    Worker function to generate edges for a single component.

    Args:
        args: Tuple of (start_idx, size, d, random_seed)

    Returns:
        Tuple of (local_adjacency, local_degrees) for this component
    """
    start_idx, size, d, seed = args

    # Set random seed for this worker (ensures reproducibility)
    random.seed(seed)

    # Create local adjacency matrix for this component
    local_adj = np.zeros((size, d), dtype=np.int32)
    local_degrees = np.zeros(size, dtype=np.int32)

    if size < 2:
        return local_adj, local_degrees

    # Local vertex indices (0 to size-1)
    local_vertices = list(range(size))

    # Connect component using local indices, then convert to global
    _connect_component_local(local_vertices, local_adj, local_degrees, d, start_idx)

    return local_adj, local_degrees


def _connect_component_local(
    vertices: list,
    adjacency: np.ndarray,
    degrees: np.ndarray,
    d: int,
    global_offset: int,
):
    """
    Connect vertices within a component (local indexing version for parallel processing).
    Stores global vertex indices (1-indexed) in the adjacency matrix.

    Args:
        vertices: List of local vertex indices (0 to len-1)
        adjacency: Local adjacency array for this component
        degrees: Local degree array for this component
        d: Degree bound
        global_offset: Offset to convert local index to global index
    """
    if len(vertices) < 2:
        return

    # Shuffle to randomize structure
    shuffled = vertices.copy()
    random.shuffle(shuffled)

    # For d == 1, we can only form pairs (limited connectivity)
    if d == 1:
        for i in range(0, len(shuffled) - 1, 2):
            _add_edge_local(
                shuffled[i], shuffled[i + 1], adjacency, degrees, d, global_offset
            )
        return

    # For d >= 2, create a path through all vertices first
    for i in range(len(shuffled) - 1):
        success = _add_edge_local(
            shuffled[i], shuffled[i + 1], adjacency, degrees, d, global_offset
        )
        if not success:
            for j in range(i):
                if degrees[shuffled[j]] < d and degrees[shuffled[i + 1]] < d:
                    if _add_edge_local(
                        shuffled[j],
                        shuffled[i + 1],
                        adjacency,
                        degrees,
                        d,
                        global_offset,
                    ):
                        break

    # Add additional random edges
    extra_edges = min(
        len(vertices) * (d - 2) // 2, len(vertices) * (len(vertices) - 1) // 4
    )
    extra_edges = max(0, extra_edges)

    attempts = 0
    max_attempts = extra_edges * 5

    while extra_edges > 0 and attempts < max_attempts:
        attempts += 1
        u = random.choice(vertices)
        v = random.choice(vertices)

        if u != v and _add_edge_local(u, v, adjacency, degrees, d, global_offset):
            extra_edges -= 1


def _add_edge_local(
    u: int,
    v: int,
    adjacency: np.ndarray,
    degrees: np.ndarray,
    d: int,
    global_offset: int,
) -> bool:
    """
    Add an undirected edge between local vertices u and v.
    Stores global vertex indices (1-indexed) in adjacency matrix.

    Args:
        u, v: Local vertex indices
        adjacency: Local adjacency array
        degrees: Local degree array
        d: Degree bound
        global_offset: Offset to convert local to global index

    Returns:
        True if edge was added, False otherwise
    """
    if degrees[u] >= d or degrees[v] >= d:
        return False

    # Global indices (1-indexed for storage)
    v_global_stored = (v + global_offset) + 1
    u_global_stored = (u + global_offset) + 1

    # Check if edge already exists
    if v_global_stored in adjacency[u, : degrees[u]]:
        return False

    # Add edge in both directions (storing global 1-indexed)
    adjacency[u, degrees[u]] = v_global_stored
    adjacency[v, degrees[v]] = u_global_stored
    degrees[u] += 1
    degrees[v] += 1

    return True


def _distribute_vertices_to_components(n: int, c: int) -> list:
    """
    Distribute n vertices among c components using a power-law distribution.
    This creates a realistic skew where isolated vertices and small components
    are much more common than large clusters.

    The distribution follows a Zipf-like pattern where the k-th largest
    component has size proportional to 1/k^alpha.
    """
    if c == 1:
        return [n]

    if c >= n:
        # More components than vertices: each gets 1 (or 0 for excess)
        sizes = [1] * n + [0] * (c - n)
        random.shuffle(sizes)
        return sizes

    # Power-law exponent - higher values create more skew toward small components
    # alpha=1.0 gives Zipf's law, alpha=1.5-2.0 gives heavier tail of small components
    alpha = 1.5

    # Generate raw sizes using power-law (Zipf-like) distribution
    # Component ranks 1, 2, 3, ... get sizes proportional to 1/rank^alpha
    ranks = np.arange(1, c + 1, dtype=np.float64)
    raw_sizes = 1.0 / np.power(ranks, alpha)

    # Normalize to sum to n, ensuring minimum of 1 per component
    raw_sizes = raw_sizes / raw_sizes.sum() * (n - c)  # Distribute n-c extra vertices
    sizes = np.floor(raw_sizes).astype(int) + 1  # Each component gets at least 1

    # Distribute remaining vertices (due to floor) to random components
    remaining = n - sizes.sum()
    if remaining > 0:
        # Add remaining to random components, preferring smaller ones
        indices = list(range(c))
        for _ in range(remaining):
            # Weight toward smaller components for additional vertices
            weights = [1.0 / (s + 1) for s in sizes]
            idx = random.choices(indices, weights=weights)[0]
            sizes[idx] += 1
    elif remaining < 0:
        # Remove excess from largest components
        while remaining < 0:
            max_idx = np.argmax(sizes)
            if sizes[max_idx] > 1:
                sizes[max_idx] -= 1
                remaining += 1
            else:
                break

    # Shuffle so largest components aren't always first
    sizes_list = sizes.tolist()
    random.shuffle(sizes_list)

    return sizes_list


def _connect_component(
    vertices: list, adjacency: np.ndarray, degrees: np.ndarray, d: int
):
    """
    Connect vertices within a component to form a connected subgraph.
    Uses a path-based spanning tree first (guarantees connectivity with d >= 2),
    then adds random edges.
    """
    if len(vertices) < 2:
        return

    # Shuffle to randomize structure
    shuffled = vertices.copy()
    random.shuffle(shuffled)

    # For d == 1, we can only form pairs (limited connectivity)
    if d == 1:
        # With degree bound 1, we can only connect pairs
        for i in range(0, len(shuffled) - 1, 2):
            _add_edge(shuffled[i], shuffled[i + 1], adjacency, degrees, d)
        return

    # For d >= 2, create a path through all vertices first (guaranteed to work)
    # This ensures every vertex has degree at most 2 from the spanning tree
    for i in range(len(shuffled) - 1):
        success = _add_edge(shuffled[i], shuffled[i + 1], adjacency, degrees, d)
        if not success:
            # This should not happen with d >= 2 for a path, but handle it
            # Try to find an alternative connection
            for j in range(i):
                if degrees[shuffled[j]] < d and degrees[shuffled[i + 1]] < d:
                    if _add_edge(shuffled[j], shuffled[i + 1], adjacency, degrees, d):
                        break

    # Add additional random edges (respecting degree bound)
    # Target: add roughly (d-2) * len(vertices) / 2 extra edges (since path uses ~2 per vertex)
    extra_edges = min(
        len(vertices) * (d - 2) // 2, len(vertices) * (len(vertices) - 1) // 4
    )
    extra_edges = max(0, extra_edges)

    attempts = 0
    max_attempts = extra_edges * 5

    while extra_edges > 0 and attempts < max_attempts:
        attempts += 1
        u = random.choice(vertices)
        v = random.choice(vertices)

        if u != v and _add_edge(u, v, adjacency, degrees, d):
            extra_edges -= 1


def _add_edge(
    u: int, v: int, adjacency: np.ndarray, degrees: np.ndarray, d: int
) -> bool:
    """
    Add an undirected edge between vertices u and v.
    Returns True if edge was added, False if not possible (degree bound or already exists).
    """
    # Check degree bounds
    if degrees[u] >= d or degrees[v] >= d:
        return False

    # Check if edge already exists (vertices stored as 1-indexed)
    v_stored = v + 1  # 1-indexed
    u_stored = u + 1  # 1-indexed

    if v_stored in adjacency[u, : degrees[u]]:
        return False

    # Add edge in both directions
    adjacency[u, degrees[u]] = v_stored
    adjacency[v, degrees[v]] = u_stored
    degrees[u] += 1
    degrees[v] += 1

    return True


def load_graph(name: str) -> tuple:
    """
    Load a stored graph.

    Args:
        name: Name of the graph file (without extension)

    Returns:
        Tuple of (adjacency, n, d, c, component_sizes) where:
            - adjacency: np.ndarray of shape (n, d), 1-indexed neighbors (0 = no neighbor)
            - n: number of vertices
            - d: degree bound
            - c: number of components
            - component_sizes: list of component sizes (sorted small to large)
    """
    filepath = GRAPHS_PATH / f"{name}.npz"
    data = np.load(filepath)
    adjacency = data["adjacency"]
    metadata = data["metadata"]
    n, d, c = int(metadata[0]), int(metadata[1]), int(metadata[2])

    # Load component sizes if available, otherwise calculate them
    if "component_sizes" in data:
        component_sizes = sorted(data["component_sizes"].tolist())
    else:
        # Calculate component sizes via BFS for backwards compatibility
        component_sizes = _calculate_component_sizes(adjacency, n)

    return adjacency, n, d, c, component_sizes


def _calculate_component_sizes(adjacency: np.ndarray, n: int) -> list:
    """
    Calculate component sizes by running BFS from unvisited vertices.

    Args:
        adjacency: The adjacency array
        n: Number of vertices

    Returns:
        List of component sizes, sorted from small to large
    """
    from collections import deque

    visited = set()
    component_sizes = []

    for start in range(n):
        if start in visited:
            continue

        # BFS from this vertex
        queue = deque([start])
        visited.add(start)
        size = 0

        while queue:
            v = queue.popleft()
            size += 1
            neighbors = get_neighbors(adjacency, v)
            for u in neighbors:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)

        component_sizes.append(size)

    return sorted(component_sizes)


def get_neighbors(adjacency: np.ndarray, v: int) -> list:
    """
    Get the neighbors of vertex v (0-indexed).

    Args:
        adjacency: The adjacency array from load_graph
        v: Vertex index (0-indexed)

    Returns:
        List of neighbor indices (0-indexed)
    """
    neighbors = adjacency[v]
    # Filter out 0s (no neighbor) and convert to 0-indexed
    return [int(u) - 1 for u in neighbors if u > 0]


def get_degree(adjacency: np.ndarray, v: int) -> int:
    """
    Get the degree of vertex v.

    Args:
        adjacency: The adjacency array from load_graph
        v: Vertex index (0-indexed)

    Returns:
        Degree of vertex v
    """
    return int(np.count_nonzero(adjacency[v]))


def query_neighbor(adjacency: np.ndarray, v: int, i: int) -> Optional[int]:
    """
    Query the i-th neighbor of vertex v (as defined in the bounded-degree model).
    This is the f_G(v, i) function from the paper.

    Args:
        adjacency: The adjacency array from load_graph
        v: Vertex index (0-indexed)
        i: Neighbor index (1-indexed as per paper, i ∈ [d])

    Returns:
        Neighbor vertex index (0-indexed) or None if v has less than i neighbors
    """
    if i < 1 or i > adjacency.shape[1]:
        return None
    neighbor = adjacency[v, i - 1]
    if neighbor == 0:
        return None
    return int(neighbor) - 1


def create_visual_graph_representation(name: str) -> Path:
    """
    Create a visual PDF representation of a stored graph.
    Components are visually separated with clear spacing if
    SEPARATE_COMPONENTS_IN_VISUALS is True.

    Args:
        name: Name of the graph file (without extension)

    Returns:
        Path to the saved PDF file
    """
    # Load the graph
    adjacency, n, d, c = load_graph(name)

    # Build NetworkX graph from adjacency matrix
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges from adjacency list
    for v in range(n):
        for neighbor_stored in adjacency[v]:
            if neighbor_stored > 0:
                u = int(neighbor_stored) - 1  # Convert to 0-indexed
                if v < u:  # Avoid adding edges twice
                    G.add_edge(v, u)

    # Choose layout based on setting
    if SEPARATE_COMPONENTS_IN_VISUALS:
        pos = _create_separated_component_layout(G)
    else:
        pos = nx.random_layout(G)

    # Adaptive sizing based on graph size - improved for large graphs
    # Figure size scales with sqrt of n for better resolution
    fig_size = max(15, min(50, int(5 * np.sqrt(n / 100))))

    # Node size inversely proportional to n, but with floor
    node_size = max(0.5, min(20, 2000 / n))

    # Edge width very thin for large graphs
    edge_width = max(0.001, min(0.5, 5 / n))

    # Edge alpha very faint for large graphs to avoid black blobs
    edge_alpha = max(0.01, min(0.5, 20 / n))

    # DPI scales up for larger graphs to maintain sharpness when zooming
    dpi = min(600, max(300, int(150 + n / 500)))

    plt.figure(figsize=(fig_size, fig_size))

    # Draw EDGES (very faint)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=edge_alpha)

    # Draw NODES (bright + visible)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color="blue",
        edgecolors="none",  # Remove node borders for cleaner look at scale
        linewidths=0,
    )

    plt.axis("off")

    # Save to VISUALS_PATH with higher DPI for large graphs
    filepath = VISUALS_PATH / f"{name}.pdf"
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()

    return filepath


def _create_separated_component_layout(G: nx.Graph) -> dict:
    """
    Create a layout where connected components are spatially separated.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping node -> (x, y) position
    """
    components = list(nx.connected_components(G))
    num_components = len(components)

    if num_components == 0:
        return {}

    n = G.number_of_nodes()

    if num_components == 1:
        # Single component, use random layout (no scipy dependency)
        # Scale based on number of nodes for better spacing
        scale = max(1.0, np.sqrt(n) / 10)
        pos = nx.random_layout(G, seed=42)
        return {node: coord * scale for node, coord in pos.items()}

    # Sort components by size (largest first) for better arrangement
    components = sorted(components, key=len, reverse=True)

    # Calculate grid dimensions for component placement
    grid_cols = int(np.ceil(np.sqrt(num_components)))

    # Threshold for using spring layout (avoid scipy for large components)
    SPRING_LAYOUT_THRESHOLD = 500

    # Calculate component sizes to determine appropriate spacing
    component_sizes = [len(c) for c in components]
    max_comp_size = max(component_sizes)
    COMPONENT_SPACING = 0.5

    # Base spacing scales with the largest component size
    # This ensures components don't overlap
    base_spacing = max(10.0, np.sqrt(max_comp_size) / 4)

    # Additional spacing factor based on total graph size
    spacing = base_spacing * max(1.1, np.log10(n + 1)) * COMPONENT_SPACING

    pos = {}

    for idx, component in enumerate(components):
        # Grid position for this component
        row = idx // grid_cols
        col = idx % grid_cols

        # Create subgraph for this component
        subgraph = G.subgraph(component)

        # Choose layout based on component size
        comp_size = len(component)
        if comp_size == 1:
            # Single node - place at center of its cell
            sub_pos = {list(component)[0]: np.array([0.0, 0.0])}
        elif comp_size == 2:
            # Two nodes - place horizontally
            nodes = list(component)
            sub_pos = {nodes[0]: np.array([-0.3, 0.0]), nodes[1]: np.array([0.2, 0.0])}
        elif comp_size <= SPRING_LAYOUT_THRESHOLD:
            # Use spring layout for smaller components (no scipy needed for small graphs)
            sub_pos = nx.spring_layout(subgraph, seed=42 + idx, iterations=50)
        else:
            # Use random layout for large components to avoid scipy dependency
            sub_pos = nx.random_layout(subgraph, seed=42 + idx)

        # Scale component layout based on its size relative to max
        # Smaller components get proportionally smaller space
        scale = max(0.5, np.sqrt(comp_size) / 2)

        # Calculate center offset for this component's grid cell
        center_x = col * spacing
        center_y = -row * spacing  # Negative to go top-to-bottom

        # Apply scaling and translation to component positions
        for node, (x, y) in sub_pos.items():
            pos[node] = np.array([center_x + x * scale, center_y + y * scale])

    return pos


def run_demo():
    """
    Run a demo to create, store, load, and visualize graphs.
    """
    # Example usage: create and store a graph
    n = 1000  # number of vertices
    d = 10  # degree bound
    c = 5  # number of components
    name = "graph_1000_10_5"

    graph_path = create_and_store_graph(n, d, c, name)
    print(f"Graph stored at: {graph_path}")

    # Load the graph
    adjacency, n_loaded, d_loaded, c_loaded = load_graph(name)
    print(
        f"Loaded graph with {n_loaded} vertices, degree bound {d_loaded}, {c_loaded} components."
    )

    # Query neighbors of vertex 0
    neighbors_v0 = get_neighbors(adjacency, 0)
    print(f"Neighbors of vertex 0: {neighbors_v0}")

    # Query degree of vertex 0
    degree_v0 = get_degree(adjacency, 0)
    print(f"Degree of vertex 0: {degree_v0}")

    # Query the 1st neighbor of vertex 0
    first_neighbor_v0 = query_neighbor(adjacency, 0, 1)
    print(f"1st neighbor of vertex 0: {first_neighbor_v0}")

    # Create visual representation
    visual_path = create_visual_graph_representation(name)
    print(f"Visual representation saved at: {visual_path}")


if __name__ == "__main__":

    # Uncomment to run demo
    # run_demo()

    create_and_store_graph(2000000, 20, 5, "graph_2000000_20_5")
    create_and_store_graph(2000000, 20, 20, "graph_2000000_20_20")
    # create_and_store_graph(10000000, 20, 1, "graph_10000000_20_1")

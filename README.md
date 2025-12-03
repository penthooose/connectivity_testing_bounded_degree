# Connectivity Testing (Bounded-Degree Model)
---
Related paper: [Connectivity Testing in the Bounded-Degree Model — Algorithms, Implementation, and Analysis](docs/Connectivity_Testing_in_the_Bounded_Degree_Model-Algorithms_Implementation_and_Analysis.pdf) — the PDF in the `docs/` folder describes the algorithms, implementation details, and analysis accompanying this repository.

````

## Run the testers

- The example in main function executes three testers defined in `Implementation/connectivity_testers.py`:

  - `basic_BFS_connectivity_tester(name)` — full BFS, O(n) (baseline)
  - `simple_connectivity_tester(name)` — Algorithm 4.1 (epsilon-far tester)
  - `improved_connectivity_tester(name)` — Algorithm 4.2 (interval-based epsilon-far tester)

- You can also call any tester directly from a Python session:

```shell
python -c "from connectivity_testers import simple_connectivity_tester; simple_connectivity_tester('graph_1000_10_5')"
````

- Each tester prints results, timings and a short analysis to stdout.

## Create and store graphs

- Graph utilities live in `graphs_helpers.py`.
- To create a bounded-degree graph programmatically, use:

```python
from graphs_helpers import create_and_store_graph
create_and_store_graph(n=1000, d=10, c=5, name='graph_1000_10_5')
```

- Graph files are saved as compressed `.npz` files under `stored_graphs/`.
- Visual PDF exports (optional) are written to `stored_visuals/` and can be created with `create_visual_graph_representation(name)`.

## Visualize graphs (create PDF)

- To generate a visual PDF for a stored graph, call:

```python
from graphs_helpers import create_visual_graph_representation
create_visual_graph_representation('graph_1000_10_5')
```

- The function builds a NetworkX representation and writes a PDF named `<name>.pdf` to `stored_visuals/`.
- Layout and appearance are controlled in `graphs_helpers.py` by flags and parameters such as `SEPARATE_COMPONENTS_IN_VISUALS`, figure sizing and DPI. Adjust those constants if you need different spacing, node sizes or resolution.

- Example one-liner (from shell) to create a PDF for an existing graph file:

```shell
python -c "from graphs_helpers import create_visual_graph_representation; print(create_visual_graph_representation('graph_1000_10_5'))"
```

## Important implementation notes

- The testers use cached loading helpers: `get_cached_graph(name)` and `clear_graph_cache(name)` in `graphs_helpers.py`.
- Graph adjacency format: stored as an `(n, d)` numpy array with 1-indexed neighbor entries (0 means no neighbor). Use helper functions `get_neighbors`, `query_neighbor`, and `get_degree` to access graph data safely.
- The `CONFIG` dictionary in `connectivity_testers.py` controls `epsilon` and algorithm constants — edit there to tune behavior.

## Algorithm provenance

The implemented testers are based on the algorithms from:

Arnab Bhattacharyya and Yuichi Yoshida — "Graphs in the Bounded-Degree Model".

(See docstrings in `connectivity_testers.py` for which algorithm corresponds to which function.)

## Minimal troubleshooting

- If a tester doesn't pick up a graph, check that `stored_graphs/<name>.npz` exists.
- If you change graph files while Python is running, call `clear_graph_cache(name=None)` to clear the in-memory cache.

---

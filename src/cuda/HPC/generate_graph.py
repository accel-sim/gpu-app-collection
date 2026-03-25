#!/usr/bin/env python3
"""
Generate a synthetic graph with ~100K vertices for benchmarking.
Creates a scale-free graph using the Barabási-Albert model.
"""

import sys
import random
import argparse


def create_barabasi_albert_graph(n, m, seed=42):
    """
    Create a Barabási-Albert scale-free graph using optimized approach.

    Args:
        n: Number of vertices
        m: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility

    Returns:
        List of edges (u, v) tuples
    """
    random.seed(seed)

    print(f"Generating Barabási-Albert graph with {n:,} vertices...")
    print(f"Each new node connects to {m} existing nodes")

    # Edge list
    edges = []

    # Targets for preferential attachment (repeating nodes based on degree)
    # This allows O(1) random selection with degree-based probability
    targets = []

    # Start with a small complete graph
    initial_nodes = max(m, 2)
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            edges.append((i, j))
            targets.append(i)
            targets.append(j)

    # Add remaining nodes with preferential attachment
    for new_node in range(initial_nodes, n):
        # Sample m unique nodes from targets (with replacement conceptually,
        # but we ensure uniqueness)
        selected = set()

        # Try to select m unique targets
        attempts = 0
        while len(selected) < m and attempts < m * 20:
            target = random.choice(targets)
            selected.add(target)
            attempts += 1

        # If we couldn't get m unique targets (very unlikely), fill with any nodes
        if len(selected) < m:
            available = set(range(new_node)) - selected
            needed = m - len(selected)
            selected.update(random.sample(list(available), min(needed, len(available))))

        # Add edges to selected nodes
        for target in selected:
            edges.append((new_node, target))
            # Add both endpoints to targets for preferential attachment
            targets.append(new_node)
            targets.append(target)

        # Progress indicator
        if (new_node + 1) % 10000 == 0:
            print(f"  Generated {new_node + 1:,} / {n:,} vertices...")

    n_edges = len(edges)
    print(f"Generated graph: {n:,} vertices, {n_edges:,} edges")
    print(f"Average degree: {2 * n_edges / n:.2f}")

    return edges, n


def save_graph_as_mtx(edges, n_vertices, output_file):
    """
    Save graph in Matrix Market (.mtx) format.

    Args:
        edges: List of (u, v) tuples
        n_vertices: Number of vertices
        output_file: Output file path
    """
    print(f"Saving to {output_file}...")

    # Make edges symmetric (undirected graph) and unique
    all_edges = set()
    for u, v in edges:
        # Ensure u < v for consistent ordering
        if u > v:
            u, v = v, u
        all_edges.add((u, v))

    # Convert to sorted list
    all_edges = sorted(all_edges)

    with open(output_file, 'w') as f:
        # Write Matrix Market header
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"% Barabási-Albert graph: {n_vertices} vertices, {len(all_edges)} edges\n")

        # Write dimensions: rows cols non-zeros
        # For an adjacency matrix, we only store the upper triangle
        f.write(f"{n_vertices} {n_vertices} {len(all_edges)}\n")

        # Write edges (1-based indexing in MTX format)
        for u, v in all_edges:
            f.write(f"{u+1} {v+1}\n")

    print(f"Successfully saved graph to {output_file}")

    # Print file size
    import os
    file_size = os.path.getsize(output_file)
    if file_size < 1024*1024:
        print(f"File size: {file_size/1024:.1f} KB")
    else:
        print(f"File size: {file_size/(1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a synthetic graph for benchmarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-n', '--vertices',
        type=int,
        default=100000,
        help='Number of vertices'
    )
    parser.add_argument(
        '-m', '--edges-per-node',
        type=int,
        default=5,
        help='Number of edges each new node attaches to (controls graph density)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='synthetic_100k.mtx',
        help='Output file name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.vertices < 1:
        print("Error: Number of vertices must be positive")
        sys.exit(1)

    if args.edges_per_node < 1 or args.edges_per_node >= args.vertices:
        print(f"Error: edges-per-node must be between 1 and {args.vertices-1}")
        sys.exit(1)

    # Generate graph
    edges, n_vertices = create_barabasi_albert_graph(
        args.vertices,
        args.edges_per_node,
        args.seed
    )

    # Save to MTX format
    save_graph_as_mtx(edges, n_vertices, args.output)

    n_edges = len(edges)
    print("\nGraph statistics:")
    print(f"  Vertices: {n_vertices:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Average degree: {2 * n_edges / n_vertices:.2f}")
    density = (2 * n_edges) / (n_vertices * (n_vertices - 1))
    print(f"  Density: {density:.6f}")

    print(f"\nUsage:")
    print(f"  BFS: cd bfs && ./run.sh --file ../{args.output}")
    print(f"  MST: cd mst && ./run.sh --file ../{args.output}")


if __name__ == '__main__':
    main()

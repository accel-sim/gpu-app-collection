/*
 * Standalone BFS kernel extracted from cugraph test suite
 * Simplified to run without gtest framework
 */

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <optional>
#include <limits>
#include <memory>

// Simple graph loader for MTX format
template <typename vertex_t, typename edge_t>
struct SimpleGraph {
    std::vector<edge_t> offsets;
    std::vector<vertex_t> indices;
    vertex_t num_vertices;
    edge_t num_edges;
};

template <typename vertex_t, typename edge_t>
SimpleGraph<vertex_t, edge_t> load_mtx_graph(const std::string& filename) {
    SimpleGraph<vertex_t, edge_t> graph;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // Check if graph is symmetric
    bool is_symmetric = false;
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
        if (line.find("symmetric") != std::string::npos) {
            is_symmetric = true;
        }
    }

    // Read dimensions
    vertex_t num_rows, num_cols;
    edge_t num_entries;
    std::istringstream iss(line);
    iss >> num_rows >> num_cols >> num_entries;

    graph.num_vertices = std::max(num_rows, num_cols);

    // Read edges
    std::vector<std::pair<vertex_t, vertex_t>> edges;
    vertex_t src, dst;
    while (file >> src >> dst) {
        src--; dst--; // MTX is 1-indexed
        edges.push_back({src, dst});
        // For symmetric graphs, add reverse edge if not a self-loop
        if (is_symmetric && src != dst) {
            edges.push_back({dst, src});
        }
    }

    graph.num_edges = edges.size();

    // Convert to CSR format
    graph.offsets.resize(graph.num_vertices + 1, 0);

    // Count degree
    for (const auto& edge : edges) {
        graph.offsets[edge.first + 1]++;
    }

    // Prefix sum
    for (vertex_t i = 0; i < graph.num_vertices; i++) {
        graph.offsets[i + 1] += graph.offsets[i];
    }

    graph.indices.resize(edges.size());
    std::vector<edge_t> current_pos = graph.offsets;

    for (const auto& edge : edges) {
        graph.indices[current_pos[edge.first]++] = edge.second;
    }

    file.close();
    return graph;
}

int main(int argc, char** argv) {
    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    // Parse command line arguments
    std::string graph_file = "karate.mtx";
    vertex_t source = 0;
    bool use_rmat = false;
    int rmat_scale = 20;
    int rmat_edge_factor = 16;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source = std::atoi(argv[++i]);
        } else if (arg == "--rmat") {
            use_rmat = true;
        } else if (arg == "--scale" && i + 1 < argc) {
            rmat_scale = std::atoi(argv[++i]);
        } else if (arg == "--edge-factor" && i + 1 < argc) {
            rmat_edge_factor = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --file <path>       Graph file in MTX format\n"
                      << "  --source <vertex>   Source vertex for BFS (default: 0)\n"
                      << "  --rmat              Use RMAT generated graph instead of file\n"
                      << "  --scale <n>         RMAT scale parameter (default: 20)\n"
                      << "  --edge-factor <n>   RMAT edge factor (default: 16)\n"
                      << "  --help, -h          Show this help message\n";
            return 0;
        }
    }

    std::cout << "=== Standalone BFS Kernel ===" << std::endl;
    std::cout << "Source vertex: " << source << std::endl;

    if (use_rmat) {
        std::cout << "RMAT generation not supported in standalone version.\n";
        std::cout << "Please use --file option with an MTX graph file\n";
        return 1;
    }

    std::cout << "Loading graph from: " << graph_file << std::endl;

    // Initialize RMM memory pool
    auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
    auto pool_mr = std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        cuda_mr.get(), 1024 * 1024 * 1024ULL); // 1GB initial pool
    rmm::mr::set_current_device_resource(pool_mr.get());

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // Load graph
    auto h_graph = load_mtx_graph<vertex_t, edge_t>(graph_file);
    std::cout << "Loaded graph: " << h_graph.num_vertices << " vertices, "
              << h_graph.num_edges << " edges" << std::endl;

    // Copy edges to device
    rmm::device_uvector<vertex_t> d_src(h_graph.num_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst(h_graph.num_edges, handle.get_stream());

    // Extract source and destination from indices/offsets
    std::vector<vertex_t> h_src, h_dst;
    for (vertex_t v = 0; v < h_graph.num_vertices; v++) {
        for (edge_t e = h_graph.offsets[v]; e < h_graph.offsets[v + 1]; e++) {
            h_src.push_back(v);
            h_dst.push_back(h_graph.indices[e]);
        }
    }

    cudaMemcpy(d_src.data(), h_src.data(), h_src.size() * sizeof(vertex_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst.data(), h_dst.data(), h_dst.size() * sizeof(vertex_t), cudaMemcpyHostToDevice);

    std::cout << "Constructing cugraph..." << std::endl;
    hr_timer.start("Graph construction");

    // Build graph from edge list
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map{std::nullopt};

    auto [graph, edge_properties, renumber_map] =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
            handle,
            std::nullopt,  // vertex list
            std::move(d_src),
            std::move(d_dst),
            std::vector<cugraph::arithmetic_device_uvector_t>{},  // no edge properties
            cugraph::graph_properties_t{true, false},  // undirected, no multi-edges
            true);  // renumber

    auto graph_view = graph.view();

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    double graph_time = hr_timer.stop();
    std::cout << "Graph construction: " << (graph_time * 1000.0) << " ms" << std::endl;

    std::cout << "Running BFS from source " << source << "..." << std::endl;

    // Allocate output
    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(), handle.get_stream());

    hr_timer.start("BFS");
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    rmm::device_scalar<vertex_t> const d_source(source, handle.get_stream());

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.data(),
                 d_predecessors.data(),
                 d_source.data(),
                 size_t{1},
                 false,  // direction optimizing (false for now)
                 std::numeric_limits<vertex_t>::max());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    double bfs_time = hr_timer.stop();
    std::cout << "BFS execution: " << (bfs_time * 1000.0) << " ms" << std::endl;

    // Copy results back
    std::vector<vertex_t> h_distances(graph_view.number_of_vertices());
    std::vector<vertex_t> h_predecessors(graph_view.number_of_vertices());

    cudaMemcpy(h_distances.data(), d_distances.data(),
               h_distances.size() * sizeof(vertex_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_predecessors.data(), d_predecessors.data(),
               h_predecessors.size() * sizeof(vertex_t), cudaMemcpyDeviceToHost);

    // Print results summary
    std::cout << "\n=== BFS Results ===" << std::endl;
    std::cout << "First 10 vertices:" << std::endl;
    std::cout << "Vertex\tDistance\tPredecessor" << std::endl;
    for (int i = 0; i < std::min(10, (int)h_distances.size()); i++) {
        std::cout << i << "\t" << h_distances[i] << "\t\t";
        if (h_predecessors[i] == cugraph::invalid_vertex_id<vertex_t>::value) {
            std::cout << "None";
        } else {
            std::cout << h_predecessors[i];
        }
        std::cout << std::endl;
    }

    // Count reachable vertices
    int reachable = 0;
    for (auto d : h_distances) {
        if (d != std::numeric_limits<vertex_t>::max()) reachable++;
    }
    std::cout << "\nReachable vertices: " << reachable << " / " << h_distances.size() << std::endl;

    std::cout << "\n=== BFS Complete ===" << std::endl;

    return 0;
}

/*
 * Standalone MST (Minimum Spanning Tree) kernel
 * Extracted and simplified from cugraph test suite
 */

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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
template <typename vertex_t, typename edge_t, typename weight_t>
struct SimpleWeightedGraph {
    std::vector<vertex_t> row_indices;
    std::vector<vertex_t> col_indices;
    std::vector<weight_t> weights;
    vertex_t num_vertices;
    edge_t num_edges;
};

template <typename vertex_t, typename edge_t, typename weight_t>
SimpleWeightedGraph<vertex_t, edge_t, weight_t> load_weighted_mtx_graph(const std::string& filename) {
    SimpleWeightedGraph<vertex_t, edge_t, weight_t> graph;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // Skip comments and read header
    std::string line;
    bool is_symmetric = false;
    bool is_pattern = false;

    while (std::getline(file, line)) {
        if (line[0] != '%') break;
        if (line.find("symmetric") != std::string::npos) {
            is_symmetric = true;
        }
        if (line.find("pattern") != std::string::npos) {
            is_pattern = true;
        }
    }

    // Read dimensions from the first non-comment line
    vertex_t num_rows, num_cols;
    edge_t num_entries;
    std::istringstream iss(line);
    iss >> num_rows >> num_cols >> num_entries;

    graph.num_vertices = std::max(num_rows, num_cols);

    // Read edges
    vertex_t src, dst;
    weight_t weight;

    while (file >> src >> dst) {
        src--; dst--; // MTX is 1-indexed

        if (is_pattern) {
            weight = 1.0; // Default weight for pattern matrices
        } else {
            file >> weight;
        }

        graph.row_indices.push_back(src);
        graph.col_indices.push_back(dst);
        graph.weights.push_back(weight);

        // For symmetric graphs, add reverse edge if not a self-loop
        if (is_symmetric && src != dst) {
            graph.row_indices.push_back(dst);
            graph.col_indices.push_back(src);
            graph.weights.push_back(weight);
        }
    }

    graph.num_edges = graph.row_indices.size();

    file.close();
    return graph;
}

// Convert COO to CSR format
template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRGraph {
    std::vector<edge_t> offsets;
    std::vector<vertex_t> indices;
    std::vector<weight_t> weights;
    vertex_t num_vertices;
    edge_t num_edges;
};

template <typename vertex_t, typename edge_t, typename weight_t>
CSRGraph<vertex_t, edge_t, weight_t> coo_to_csr(const SimpleWeightedGraph<vertex_t, edge_t, weight_t>& coo) {
    CSRGraph<vertex_t, edge_t, weight_t> csr;
    csr.num_vertices = coo.num_vertices;
    csr.num_edges = coo.num_edges;

    // Initialize offsets
    csr.offsets.resize(csr.num_vertices + 1, 0);

    // Count degree
    for (const auto& src : coo.row_indices) {
        csr.offsets[src + 1]++;
    }

    // Prefix sum
    for (vertex_t i = 0; i < csr.num_vertices; i++) {
        csr.offsets[i + 1] += csr.offsets[i];
    }

    // Fill indices and weights
    csr.indices.resize(coo.num_edges);
    csr.weights.resize(coo.num_edges);
    std::vector<edge_t> current_pos = csr.offsets;

    for (size_t i = 0; i < coo.row_indices.size(); i++) {
        vertex_t src = coo.row_indices[i];
        edge_t pos = current_pos[src]++;
        csr.indices[pos] = coo.col_indices[i];
        csr.weights[pos] = coo.weights[i];
    }

    return csr;
}

int main(int argc, char** argv) {
    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    // Parse command line arguments
    std::string graph_file = "graphs/karate.mtx";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --file <path>    Graph file in MTX format (default: graphs/karate.mtx)\n"
                      << "  --help, -h       Show this help message\n";
            return 0;
        }
    }

    std::cout << "=== Standalone MST (Minimum Spanning Tree) Kernel ===" << std::endl;
    std::cout << "Loading graph from: " << graph_file << std::endl;

    // Load graph
    auto coo_graph = load_weighted_mtx_graph<vertex_t, edge_t, weight_t>(graph_file);
    std::cout << "Loaded graph: " << coo_graph.num_vertices << " vertices, "
              << coo_graph.num_edges << " edges" << std::endl;

    // Convert to CSR
    std::cout << "Converting to CSR format..." << std::endl;
    auto csr_graph = coo_to_csr(coo_graph);

    // Initialize RMM memory resource
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 512 * 1024 * 1024ULL);  // 512 MB pool
    rmm::mr::set_current_device_resource(&pool_mr);

    // Create RAFT handle
    raft::handle_t handle;

    // Copy graph to device
    std::cout << "Copying graph to device..." << std::endl;
    rmm::device_uvector<edge_t> d_offsets(csr_graph.offsets.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_indices(csr_graph.indices.size(), handle.get_stream());
    rmm::device_uvector<weight_t> d_weights(csr_graph.weights.size(), handle.get_stream());

    raft::update_device(d_offsets.data(), csr_graph.offsets.data(),
                       csr_graph.offsets.size(), handle.get_stream());
    raft::update_device(d_indices.data(), csr_graph.indices.data(),
                       csr_graph.indices.size(), handle.get_stream());
    raft::update_device(d_weights.data(), csr_graph.weights.data(),
                       csr_graph.weights.size(), handle.get_stream());

    // Create cuGraph CSR view
    std::cout << "Constructing cugraph..." << std::endl;
    cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> graph_view(
        d_offsets.data(),
        d_indices.data(),
        d_weights.data(),
        csr_graph.num_vertices,
        csr_graph.num_edges);

    handle.sync_stream();

    // Run MST
    std::cout << "Running MST algorithm..." << std::endl;

    HighResTimer hr_timer{};
    hr_timer.start("MST");

    auto mst_edges = cugraph::minimum_spanning_tree<vertex_t, edge_t, weight_t>(handle, graph_view);

    handle.sync_stream();
    hr_timer.stop();

    std::cout << "\n=== MST Results ===" << std::endl;
    hr_timer.display_and_clear(std::cout);

    // Calculate MST weight
    auto mst_weight = thrust::reduce(
        thrust::device_pointer_cast(mst_edges->view().edge_data),
        thrust::device_pointer_cast(mst_edges->view().edge_data) + mst_edges->view().number_of_edges);

    auto total_weight = thrust::reduce(
        thrust::device_pointer_cast(d_weights.data()),
        thrust::device_pointer_cast(d_weights.data()) + csr_graph.num_edges);

    std::cout << "MST edges: " << mst_edges->view().number_of_edges << std::endl;
    std::cout << "MST total weight: " << mst_weight << std::endl;
    std::cout << "Original graph total weight: " << total_weight << std::endl;
    std::cout << "MST weight ratio: " << (mst_weight / total_weight * 100.0) << "%" << std::endl;
    std::cout << "\nExpected MST edges for " << csr_graph.num_vertices
              << " vertices: " << (csr_graph.num_vertices - 1) << std::endl;

    std::cout << "\n=== MST Complete ===" << std::endl;

    return 0;
}

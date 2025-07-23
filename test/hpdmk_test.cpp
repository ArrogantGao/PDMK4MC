#include <gtest/gtest.h>
#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

#include <tree.hpp>
#include <sctl.hpp>

// TEST(HPDMKTest, BasicAssertions) {
//     hpdmk::HPDMKParams<double> params;
//     params.n_per_leaf = 20;
//     params.eps = 1e-4;
//     params.L = 10.0;

//     sctl::Vector<double> r_src(1000);
//     sctl::Vector<double> charge(1000);

//     hpdmk::HPDMKPtTree<double> tree(sctl::Comm::Self(), params, r_src, charge);
// }

void test_tree() {
    HPDMKParams params;
    params.n_per_leaf = 20;
    params.eps = 1e-4;
    params.L = 100.0;

    omp_set_num_threads(16);
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;

    int n_src = 10000;
    double r_src[n_src * 3];
    double charge[n_src];

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);
    std::uniform_real_distribution<double> distribution_charge(-1, 1);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = distribution_charge(generator);
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);

    sctl::Long i_node = tree.level_indices[1][0];
    auto &node_mid = tree.GetNodeMID();
    auto &node_attr = tree.GetNodeAttr();
    auto &node_list = tree.GetNodeLists();

    std::cout << "node " << i_node << " is at depth " << node_mid[i_node].Depth() << std::endl;

    for (int i = 0; i < 27; ++i) {
        if (node_list[i_node].nbr[i] != -1) {
            std::cout << "neighbor " << node_list[i_node].nbr[i] << " is at depth " << int(node_mid[node_list[i_node].nbr[i]].Depth()) << std::endl;
        }
    }

    sctl::Long leaf_node_i = tree.level_indices[tree.max_depth - 2][0];
    std::cout << "leaf node " << leaf_node_i << " is at depth " << int(node_mid[leaf_node_i].Depth()) << std::endl;
    for (int i = 0; i < 27; i++) {
        std::cout << "the " << i + 1 << "th neighbor of leaf node " << leaf_node_i << " is " << node_list[leaf_node_i].nbr[i] << std::endl;
    }

    std::cout << "center of node " << leaf_node_i << " is " << tree.centers[leaf_node_i * 3] << ", " << tree.centers[leaf_node_i * 3 + 1] << ", " << tree.centers[leaf_node_i * 3 + 2] << std::endl;

    for (int i = 0; i < tree.neighbors[leaf_node_i].coarsegrain.Dim(); i++) {
        std::cout << "coarsegrain neighbor " << tree.neighbors[leaf_node_i].coarsegrain[i] << " is at depth " << int(node_mid[tree.neighbors[leaf_node_i].coarsegrain[i]].Depth()) << " and center " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3] << ", " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3 + 1] << ", " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3 + 2] << std::endl;
        auto shift = tree.node_shift(leaf_node_i, tree.neighbors[leaf_node_i].coarsegrain[i]);
        std::cout << "shift from leaf node " << leaf_node_i << " to coarsegrain neighbor " << tree.neighbors[leaf_node_i].coarsegrain[i] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    }
    for (int i = 0; i < tree.neighbors[leaf_node_i].collegue.Dim(); i++) {
        std::cout << "collegue neighbor " << tree.neighbors[leaf_node_i].collegue[i] << " is at depth " << int(node_mid[tree.neighbors[leaf_node_i].collegue[i]].Depth()) << " and center " << tree.centers[tree.neighbors[leaf_node_i].collegue[i] * 3] << ", " << tree.centers[tree.neighbors[leaf_node_i].collegue[i] * 3 + 1] << ", " << tree.centers[tree.neighbors[leaf_node_i].collegue[i] * 3 + 2] << std::endl;
        auto shift = tree.node_shift(leaf_node_i, tree.neighbors[leaf_node_i].collegue[i]);
        std::cout << "shift from leaf node " << leaf_node_i << " to collegue neighbor " << tree.neighbors[leaf_node_i].collegue[i] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    }

    // auto &node_depth3 = tree.level_indices[3];
    // for (int i = 0; i < node_depth3.Dim(); i++) {
    //     auto &nbrs = node_list[node_depth3[i]].nbr;
    //     for (int j = 0; j < 27; j++) {
    //         if (nbrs[j] != -1) {
    //             auto center_xi = tree.centers[node_depth3[i] * 3];
    //             auto center_xj = tree.centers[nbrs[j] * 3];
    //             auto center_yi = tree.centers[node_depth3[i] * 3 + 1];
    //             auto center_yj = tree.centers[nbrs[j] * 3 + 1];
    //             auto center_zi = tree.centers[node_depth3[i] * 3 + 2];
    //             auto center_zj = tree.centers[nbrs[j] * 3 + 2];

    //             auto shift = tree.node_shift(node_depth3[i], nbrs[j]);
    //             std::cout << "shift from node " << node_depth3[i] << " to node " << nbrs[j] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    //             std::cout << "center of node " << node_depth3[i] << " is " << center_xi << ", " << center_yi << ", " << center_zi << std::endl;
    //             std::cout << "center of node " << nbrs[j] << " is " << center_xj << ", " << center_yj << ", " << center_zj << std::endl << std::endl;
    //         }
    //     }
    // }
    
}


int main() {

    MPI_Init(nullptr, nullptr);
    
    // hpdmk_tree_create(MPI_COMM_WORLD, params, n_src, r_src, charge);
    test_tree();

    MPI_Finalize();

    return 0;
}
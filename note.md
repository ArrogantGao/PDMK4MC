# Notes

## Features about SCTL's PtTree

1. It records all nodes of the tree in a vector, but only the leaf nodes are carrying messages.
2. Only neighbors of the same level are recorded. Coarse-grain neighbors are recorded as `-1`, one need to check their parent node.
3. If the coords recovered from the Morton code are $(x, y, z)$, then range of the node is $[x, x + dx] \times [y, y + dy] \times [z, z + dz]$.

## Data to store for computation

1. nodes to be calculated at each level `sctl::Vector<sctl::Vector<int>> level_indices`
2. neighbors of each node: including fine neighbors, collegues and coarse-grain neighbors. `sctl::Vector<NodeNeighbors> neighbors`
3. center location of each node `sctl::Vector<sctl::Vector<Real>> center`
4. plane wave coefficients, for each particle (q, x, y, z), including three different plane wave kf, kc, kcg if the node have fine, coarse-grain and collegue neighbors, calculate array of exp(ikx), exp(iky) and exp(ikz) `sctl::Vector<NodePlaneWaveCoeffs> plane_wave_coeffs`
5. interaction matrix for each level, three dimensional array of size (2 nf + 1)^3, simply stored as vectors
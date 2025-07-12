# Features about SCTL's PtTree

1. It records all nodes of the tree in a vector, but only the leaf nodes are carrying messages.
2. Only neighbors of the same level are recorded. Coarse-grain neighbors are recorded as `-1`, one need to check their parent node.
3. If the coords recovered from the Morton code are $(x, y, z)$, then range of the node is $[x, x + dx] \times [y, y + dy] \times [z, z + dz]$.
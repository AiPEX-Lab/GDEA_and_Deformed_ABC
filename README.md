# GDEA_and_Deformed_ABC

This work is under review. Please cite the published paper after it is accepted.

## Abstract
We propose iterative message passing using a global-weight matrix assembled via the direct stiffness method. This method is implemented through a deep equilibrium model with Anderson acceleration to ensure fast convergence and low memory cost. Our model is evaluated on published FEM datasets as well as a new dataset, Deformed ABC, featuring diverse geometries, materials, and boundary conditions.

## GDEA

In this work, we reformulate message passing as an iterative update using a global-weight matrix to overcome the limitations of GNNs and graph transformers. In this method, a two-by-two local weight matrix is predicted for each edge, and these matrices are then assembled into a global-weight matrix via the direct stiffness method. The global-weight matrix is used to update the node features via message passing until they converge. This approach introduces two major challenges. First, accumulating gradients over iterations can easily lead to memory shortages. Accordingly, we employ a deep equilibrium model (DEQ) to maintain constant memory usage regardless of the number of iterations. The converged equilibrium point in DEQ effectively encodes the result of an unbounded number of forward passes. Second, iterative updates to node features and the DEQ's backward solver may require excessive iterations, slowing the model. To mitigate this issue, DEQ is often combined with acceleration methods to improve convergence speed. In this study, we adopt Anderson acceleration, which leverages prior records to accelerate convergence to the solution.

An example GDEA code is provided in the GDEA folder of this repository.

## Deformed ABC

We introduce Deformed ABC, a linear-elastic FEM dataset featuring diverse geometries from the ABC dataset, randomly selected materials, and boundary conditions. Geometries are represented as a mesh graph. FEM conditions and solutions are encoded as nodal features. Each node includes (1) three dimensional node coordinates, which represents position of the node, (2) Signed Distance Field (SDF), a negative distance of the node from the surface, (3) normal vector, (4) Dirichlet boundary condition, where zero is assigned to fixed nodes and one to free nodes, (5) Nodal loads, represented as three-dimensional vectors, corresponding to the force components, (6) material data, which is Young's modulus and Poisson's ratio, and (7) displacement, which is the solution of FEM. FEM surrogate models predict nodal displacements from other node-wise input features. The resulting dataset is then preprocessed via filtering and scaling.

Further details on the FEM formulation and data preprocessing are provided in the Supplementary Information, and Code snippets for data preprocessing are available in this repository.

The original and scaled Deformed ABC is shared at huggingface.co/datasets/Junghunl/Deformed-ABC

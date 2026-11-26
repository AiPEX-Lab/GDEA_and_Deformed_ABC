# GDEA_and_Deformed_ABC

Please cite: Lee, Junghun, and Conrad Tucker. "Global weight deep equilibrium attention for numerical simulations on unstructured meshes." Communications Engineering (2026).

## Abstract
We propose iterative message passing using a global-weight matrix assembled via the direct stiffness method. We implement this method as a deep equilibrium model with Anderson acceleration to ensure fast convergence and low memory cost.

## Global weight deep equilibrium attention for numerical simulations on unstructured meshes (GDEA)

In this work, we reformulate message passing as an iterative update using a global-weight matrix to overcome the limitations of GNNs and graph transformers. In this method, we predict a two-by-two local weight matrix for each edge, then assemble these matrices into a global-weight matrix via the direct stiffness method. The global-weight matrix updates node features via message passing until convergence. This approach introduces two major challenges. First, accumulating gradients over iterations can easily lead to memory shortages. Accordingly, we employ a deep equilibrium model (DEQ) to maintain constant memory usage regardless of the number of iterations. The converged equilibrium point in DEQ effectively encodes the result of an unbounded number of forward passes. Second, iterative updates to node features and the DEQ's backward solver may require excessive iterations, slowing the model. To mitigate this issue, DEQ is often combined with acceleration methods to speed convergence. In this study, we adopt Anderson acceleration, which leverages prior records to accelerate convergence. This DEQ component models relationships between nodes. Finally, we add linear attention at the end to model feature interactions within each node.

<img width="1348" height="395" alt="1125Asset 7@2x" src="https://github.com/user-attachments/assets/5bbab10f-3aca-4507-94ab-17c57aea3908" />

An example GDEA code is provided in the GDEA folder of this repository.

## Deformed ABC

We introduce Deformed ABC, a linear-elastic FEM dataset featuring diverse geometries from the ABC dataset, randomly selected materials, and boundary conditions. We represent geometries as a mesh graph. FEM conditions and solutions are encoded as nodal features. Each node includes (1) three-dimensional node coordinates, which represent the position of the node, (2) Signed Distance Field (SDF), a negative distance of the node from the surface, (3) normal vector, (4) Dirichlet boundary condition, where zero is assigned to fixed nodes and one to free nodes, (5) Nodal loads, represented as three-dimensional vectors, corresponding to the force components, (6) material data, which are Young's modulus and Poisson's ratio, and (7) displacement, which is the solution of FEM. FEM surrogate models predict nodal displacements from other node-wise input features. We then preprocess the resulting dataset via filtering and scaling.

Further details on the FEM formulation and data preprocessing are provided in the Supplementary Material, and Code snippets for data preprocessing are available in this repository.

<img width="1869" height="831" alt="1125Asset 8@2x" src="https://github.com/user-attachments/assets/a0a8a362-439d-494a-8a84-e3d906406a0b" />

The original and scaled Deformed ABC is shared at huggingface.co/datasets/Junghunl/Deformed-ABC

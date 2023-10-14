# TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

The scale of TPU GRAPHS poses several new research challenges:
-  How to train a neural network model that can perform graph-level predictions when the memory required to train the model on a single graph may not fit on a single device?
-  How to make a model generalize well to unseen graphs when they are diverse, and the training data may be imbalanced?
-  How to improve the efficiency of a training pipeline when multiple data points contain a large amount of redundant data (same core graph but different graph configurations)?

# Glossary

**_Layout_** - collection of configurations control, how tensors are laid out in the physical memory, by specifying the dimension order of each input and output of an operation node. It contains **31** million pairs of graphs and configurations, averaging over **7,700** nodes per graph.

A _**Tile**_ collection of configuration, controls, the tile size of each fused subgraph. Contains **13** millions pairs of kernels and configurations, averaging **40** nodes per kernel subgraph

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/c264223d-c080-4d53-8d81-be3eb6aa2b2f)


# Conception

THe Dataset is a bunch of multiple collections of data, differing in terms of (1) the compiler optimization (i.e., layout and tile), (2) the source of graphs, and (3) the search strategy.

# Baseline model

**Layout.** Our default baseline model is a 3-layer GraphSAGE. We concatenate node features and per-node configuration features as inputs to the GNN. If a node is non-configurable (having no layout configuration), we use a zero vector as configuration features. 

**Tile.** For the tile collection, we implement three baselines using TensorFlow-2 and TF-GNN: an MLP model and two GNNs (GraphSAGE and GCN with residual connections). The MLP model embeds all opcodes, concatenates with node features, sums across all nodes, then concatenates with kernel configuration features, feeding into 3-layer MLP.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/1a6bb122-2860-4f18-8430-5912375e4b96)


# Improvement suggestions:

- Computation graph typicallycontains repeated subgraphs, representing repeated blocks of neural network layers. One direction is to leverage this repeated structure to devise a more compact representation that is easier to learn.

- The dataset may contain some types of graphs, e.g., ResNet, significantly more than others. This skew may make the learned model perform well on common types of graphs, but poorly on uncommon types. One may investigate how to address this data imbalance problem to improve the quality of the learned model.

- One idea is to combine the best of both worlds, using analytical modeling when easy to do and letting the learned model make corrections to the analytical estimates.

- We acknowledge that the diversity of graphs in the dataset is extremely important for the generalizability of the model. One way to generate more realistic tensor programs is to leverage Neural Architecture Search. We leave this as future work, potentially the next version of the dataset.

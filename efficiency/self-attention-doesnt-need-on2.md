**Self-attention Does Not Need O(n2) Memory**

In the paper Self-attention Does Not Need O(n2) Memory, the Google team introduces simple algorithms for attention and self-attention that require only constant memory and logarithmic memory, respectively. At a sequence length of 16384, the approach can reduce the self-attention memory overhead by 59x for inference and by 32x for differentiation

**Intro**

The result of the attention operation for a single query, is hence a weighted sum of the value vectors, where the weights
are the softmax of the dot products of the query and the keys.
The straight-forward implementation of the attention operation above requires us to first compute and remember si for
all i, leading to a O(n) time and memory complexity for each query. Transformers use self-attention, which issues a
separate query for each position in the sequence, so the overall time and space complexity is O(n**2).

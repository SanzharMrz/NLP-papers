**Self-attention Does Not Need O(n2) Memory**

In the paper Self-attention Does Not Need O(n2) Memory, the Google team introduces simple algorithms for attention and self-attention that require only constant memory and logarithmic memory, respectively. At a sequence length of 16384, the approach can reduce the self-attention memory overhead by 59x for inference and by 32x for differentiation

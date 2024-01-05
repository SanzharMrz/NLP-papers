# Self-attention Does Not Need O(n2) Memory

In the paper Self-attention Does Not Need O(n2) Memory, the Google team introduces simple algorithms for attention and self-attention that require only constant memory and logarithmic memory, respectively. At a sequence length of 16384, the approach can reduce the self-attention memory overhead by 59x for inference and by 32x for differentiation

**Intro**

<img width="632" alt="Screenshot 2024-01-04 at 14 43 40" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/a4eacf1b-8f47-48db-b826-5df5b15587ae">

attention algorithim:
1) The attention operation on a single query produces a weighted sum of value vectors.
2) The weights are determined by the softmax of the dot products between the query and the keys.

In the implementation of this we have to: first compute and remember S(i) for
all i, -> a O(n) time and memory complexity for each query.
Transformers use self-attention, it means for each element of sequence we need own query ->  time and space complexity is O(n**2).
# Main algo

## Single attention case

first step:

<img width="635" alt="Screenshot 2024-01-04 at 15 59 28" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/99f2b734-11c3-4660-9a1b-2c2427e851fd">


second step:

1) init vectors v* and scalar s* with 0

2) loop over k_n = [k1,k2,..kn] and v_n=[v1,v2,...vn] for each k_i and v_i compute s_i
   
3) for each s_i update v* and s*
   
4) after all divide v*/s* -> final result
   

<img width="640" alt="Screenshot 2024-01-04 at 15 57 59" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/fb5dc415-34fb-44f2-8473-a20b8d2e7c36">

## Self-attention case

To extend this algorithm to self-attention, just compute the results to all queries sequentially.

# Numerical stability problem

Default and new attention are not numerically stable when using floating point arithmetic
for example:
Forscores ≥ 89 the exponentiation results in inf (for bfloat16 and float32).

To resolve this problem, they invented additional scalar - m*
1) which keeps track of the maximum score that the incremental algorithm has seen so far
   
2) renormalize the sums of exponentiated values as needed
<img width="645" alt="Screenshot 2024-01-04 at 16 21 35" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/28fdf34b-0223-480b-9930-39234cc79d8d">

# Results
<img width="650" alt="Screenshot 2024-01-04 at 16 19 11" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/af6bd29e-5851-44b5-b748-ed9634ee01a0">

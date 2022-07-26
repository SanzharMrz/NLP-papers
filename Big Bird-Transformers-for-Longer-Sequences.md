Authors proposed, BIGBIRD, a sparse attention mechanism that reduces this quadratic dependency to linear. The proposed sparse attention can handle sequences
of length up to 8x of what was previously possible using similar hardware.

In particular, BIGBIRD consists of three main part:
• A set of g global tokens attending on all parts of the sequence (Like <CLS>).
• All tokens attending to a set of w local neighboring tokens.
• All tokens attending to a set of r random tokens.

<img width="1203" alt="image" src="https://user-images.githubusercontent.com/48170101/180950754-27c5a2b1-0946-4638-b201-dfc5cd1f2072.png">

  
They transform the sparse local and random attention into dense tensor operations to take full advantage of modern single instruction, multiple data (SIMD) hardware.
To do this, they first “blockify” the attention mechanism to better leverage GPUs/TPUs, which are designed to operate on blocks. Then they convert the sparse attention mechanism computation into a dense tensor product through a series of simple matrix operations such as reshape, roll, and gather, as illustrated in the animation below.

![Pipe](https://github.com/SanzharMrz/NLP-papers/blob/main/meta/image3.gif)

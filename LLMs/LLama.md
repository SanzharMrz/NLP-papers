# LLaMA: Open and Efficient Foundation Language Models

LLaMA is a collection of foundation language models ranging from 7B to 65B parameters.
They have been trained on trillions of tokens. 
Also, it shows that it is possible to train state-of-the-art models using exclusively publicly available datasets,
without resorting to proprietary and inaccessible datasets.
Large transformers were trained on a substantial amount of textual data using a standard optimizer.

# Dataset

| Dataset          | Sampling prop. | Epochs | Disk size |
|------------------|----------------|--------|-----------|
| CommonCrawl      | 67.0%          | 1.10   | 3.3 TB    |
| C4               | 15.0%          | 1.06   | 783 GB    |
| Github           | 4.5%           | 0.64   | 328 GB    |
| Wikipedia        | 4.5%           | 2.45   | 83 GB     |
| Books            | 4.5%           | 2.23   | 85 GB     |
| ArXiv            | 2.5%           | 1.06   | 92 GB     |
| StackExchange    | 2.0%           | 1.03   | 78 GB     |

# Tokenizer

The authors tokenize the data using the bytepair encoding (BPE) algorithm (Sennrich et al., 2015),
employing the implementation from SentencePiece (Kudo and Richardson, 2018).
Notably, they split all numbers into individual digits and fallback to bytes to decompose unknown UTF-8 characters.
| Params | Dimension | N Heads | N Layers | Learning Rate | Batch Size | N Tokens |
|--------|-----------|---------|----------|----------------|------------|----------|
| 6.7B   | 4096      | 32      | 32       | 3.0e−4          | 4M         | 1.0T     |
| 13.0B  | 5120      | 40      | 40       | 3.0e−4          | 4M         | 1.0T     |
| 32.5B  | 6656      | 52      | 60       | 1.5e−4          | 4M         | 1.4T     |
| 65.2B  | 8192      | 64      | 80       | 1.5e−4          | 4M         | 1.4T     |


Table 2: Model sizes, architectures, and optimization hyper-parameters.

# Architecture

The network is based on the transformer architecture (Vaswani et al., 2017). (encoder-decoder)

**Pre-normalization [GPT3].** To improve the training stability, they normalize the input of each transformer sub-layer, instead of normalizing the output(RMSNorm)

**SwiGLU activation function [PaLM].** They replace the ReLU non-linearity by the SwiGLU activation function, to improve the performance. 

**Rotary Embeddings [GPTNeo].** They remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), at each layer of the network.

```python
# Optimizer Configuration
optimizer_type = "AdamW"
hyperparameters = {
    "beta1": 0.9,
    "beta2": 0.95,
    "learning_rate_schedule": "cosine",
    "final_learning_rate_ratio": 0.1,
    "weight_decay": 0.1,
    "gradient_clipping": 1.0,
    "warmup_steps": 2000
}
```

They use an efficient
implementation of the causal multi-head attention
to reduce memory usage and runtime.(review in progress) This implementation, available in the xformers library and uses the
backward from Dao et al. (2022). This is achieved
by not storing the attention weights and not computing the key/query scores that are masked due to
the causal nature of the language modeling task.


Even they use backward pass with checkpointing, they still save the activations that
are expensive to compute, such as the outputs of
linear layers. This is achieved by manually implementing the backward function for the transformer
layers, instead of relying on the PyTorch autograd.

Also while training process authors have used the model and sequence parallelism technic, as described by
Korthikanti et al. (2022).(review in progress)


During training, They tracked the performance of their
models on a few question answering and common
sense benchmarks.
On most benchmarks, the performance improves
steadily, and correlates with the training perplexity
of the model (see Figure 1).

<img width="613" alt="Screenshot 2024-01-03 at 15 43 53" src="https://github.com/SanzharMrz/NLP-papers/assets/46630209/ecf3e8fd-a5ef-4ec2-a253-4c254a9fa8bc">

**Carbon footprint**
The training of this models have consumed a massive quantity of energy, responsible for the emission of carbon dioxide.
Authors calculated and compared the emissions of carbon for different models.
Authors hope that releasing these models will help to
reduce future carbon emission since the training is
already done, and some of the models are relatively
small and can be run on a single GPU.

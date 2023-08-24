# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

Authours propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

# Conception

For example we have one layer without non-linearity:

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/d0c63624-5dda-4958-a334-e99db5897343)

If we want to somehow modify (tune) original weights ```W``` to obtain ```W'```.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/fc35bb42-1a77-4523-8df2-9655e6638879)

This, can be interpreted as the result of another, separate, fully connected layer. We can represent the matrix ```W'``` as a product of two matrices ```A``` and ```B```, this will greatly benefit the number of parameters to be trained (for the example, we have a matrix 100 x 70, which contains 7000 numbers, and the two on the right side of the picutre: 140 + 200 = 340).
In general we will need to train in:

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/d7982a35-9863-41d6-8044-d49286d2dd3f)

fewer parameters. r is chosen to be small of order 2-8, making this value very small (approx 10^(-2)), however we lose a little in generality since now, we automatically assume that ```W'``` has a low rank.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/454a2548-8d79-41ea-9dd3-9015796b6203)

In this way, during training, we need to keep in memory the weights ```W``` of the original model and ```W'= B @ A``` of the pre-trained model, and read the gradients only for the "new" small matrices ```A``` and ```B```. They use a random Gaussian initialization for A and zero for B, so ```W'= B @ A``` is zero at the beginning of training.

# Related works

Placeholder

# Metrics

Placeholder

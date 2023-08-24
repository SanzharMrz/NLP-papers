# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

Authours propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

# Related works

- Adapter layers. Injection layers, we training them while freezing the whole network. Takes too much time for inference

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/b4baf5a6-2047-466b-b024-5daa0ea09484)

- Classical fine-tuniung. Nothing to add (a lot of computational efforts)
- Prefix-embedding tuning (PreEmbed) inserts special tokens among the input tokens. These special tokens have trainable word embeddings and are generally not in the model’s vocabulary.
- Bias-only or BitFit is a baseline where we only train the bias vectors while freezing everything else.
Contemporarily, this baseline has also been studied by BitFit
- ..
- etc

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

# Insights

- Significantly less resource-intensive pre-training. Now a model of LLaMA / GPT-3* / .... level can be retrained by any owner of a mass video card or even using google colab, from a phone)). any owner of a mass video card or in general using google colab, from a phone)))).

- Reducing the number of trained parameters reduces the requirements to the dataset.

- LoRA models take up significantly less disk space. We store one "base" model, which really weighs a lot, and a large number of LoRA modules (e.g. styles for Stable Diffusion or additional training for different languages for Copilot), which weigh almost nothing. This makes such models easier to store and distribute. For GPT-3, with 350 GB weights, the A and B matrices for all linear layers totaled 35 MB!

- No output latency. We can compute W' = W + BA before use, so a new model will require the same amount of computation as a non-faintune model.

- It is possible to change matrices A and B right on the fly, in the middle of a dialog, asking the user, for example, what style to answer in.

# Metrics

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/0aa16398-ccd8-4d49-a0be-c5ce576ab3f5)

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/028b19d0-a3d5-4605-acc1-e0513bee60ad)

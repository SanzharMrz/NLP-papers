**Pix2Struct is an image-encoder-text-decoder based on the Vision Transformer (ViT)**

To summarize, our major contributions are as follows:

• We introduce the area of general-purpose visually-situated language understanding, which consists of diverse tasks but common challenges.

• We propose a screenshot parsing pretraining objective based on the HTML source of web
pages. We show that our objective is more effective than previous attempts at enabling the
elegant pixel-to-text design for general-purpose visually-situated language understanding.

• We introduce variable-resolution input representations to the Vision Transformer and new finetuning strategies that seamlessly integrate language and vision inputs by directly rendering any
language prompts on top of the input image

**Problem with VIT**

Before extracting fixed-size patches, the standard ViT scales the input images to a
predefined resolution, which creates two undesirable effects:
(1) rescaling the image distorts the true aspect ratio, which can be highly variable for documents, mobile UIs, and figures.
(2) transferring these models to downstream tasks with higher resolution is non-trivial.
**Solution:**

1) always scale our input image up or down such that we extract the maximal
number of patches that fit within the given sequence length.
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/465f1d7e-a435-4c55-88c3-c46fed97069b)



2)to handle variable resolutions unambiguously, we use 2-dimensional absolute positional
embeddings for the input patches

**Pretraining**

Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/da10387d-bcb2-4379-a189-43ec09ca2ebc)
BART-like learning signal by masking 50% of the text while decoding the entire subtree. The masked regions are randomly sampled
spans of text from the chosen subtree where we draw crossed-out opaque bounding boxes

****warmup stage before pretraining****

if we first expose the model to a short, intense “warmup” stage of simply learning to read, we find a strong curriculum learning
effect where (1) pretraining is more stable and converges faster, and (2) we observe better finetuning
performance, 
Specifically, they create images of text snippets with random
colors and fonts on a white background. The model simply needs to decode the original text.
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/62e01e24-d839-4236-96eb-8899416ee678)

Results:

pretrain two model variants: 

(a) a base model with 282M parameters including 12
transformer layers with a hidden size of 768, and 

(b) a large model with 1.3B parameters including
18 layers with a hidden size of 1536. 

Both models have the same warmup stage using text rendered
from BooksCorpus (Zhu et al., 2015) lasting 30K steps with a maximum input sequence length of
128 patches. 

The base model  -  270K steps with the screenshot parsing
objective using

batch size -  3072 on 64 Google Cloud TPUs.

The large model -  170K steps with 

batch size of 1024 on 128 Google Cloud TPUs.

input sequence length  -  2048 patches 

optimizer -  Adafactor  

learning rate schedule uses a linear warmup of 1000 steps to 0.01, followed by cosine decay to 0.

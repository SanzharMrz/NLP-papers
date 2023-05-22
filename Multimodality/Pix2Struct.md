**Pix2Struct is an image-encoder-text-decoder based on the Vision Transformer (ViT)
Problem with VIT

Before extracting fixed-size patches, the standard ViT scales the input images to a
predefined resolution, which creates two undesirable effects:
(1) rescaling the image distorts the true aspect ratio, which can be highly variable for documents, mobile UIs, and figures.
(2) transferring these models to downstream tasks with higher resolution is non-trivial.
**Solution:

1) always scale our input image up or down such that we extract the maximal
number of patches that fit within the given sequence length
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/262341de-81d0-4b6c-875c-e909c6010e15)


2)to handle variable resolutions unambiguously, we use 2-dimensional absolute positional
embeddings for the input patches

**Pretraining

Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/da10387d-bcb2-4379-a189-43ec09ca2ebc)
BART-like learning signal by masking 50% of the text while decoding the entire subtree. The masked regions are randomly sampled
spans of text from the chosen subtree where we draw crossed-out opaque bounding boxes

****warmup stage before pretraining

if we first expose the model to a short, intense “warmup” stage of simply learning to read, we find a strong curriculum learning
effect where (1) pretraining is more stable and converges faster, and (2) we observe better finetuning
performance, 
Specifically, they create images of text snippets with random
colors and fonts on a white background. The model simply needs to decode the original text.
![image](https://github.com/SanzharMrz/NLP-papers/assets/46630209/62e01e24-d839-4236-96eb-8899416ee678)

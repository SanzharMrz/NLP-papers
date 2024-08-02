# Evolutionary Optimization of Model Merging Recipes

# About

> We present a methodology that leverages evolutionary algorithms to facilitate the merging of foundation models. Our approach is distinguished by its ability to navigate both parameter space (weights) and the data flow space (inference path), proposing a framework that integrates these two dimensions.

This paper is about using evolutionary algorithms for model merging, where merging is literally stacking / mixing weights, layers from already pretrained foundation models in different domains and languages. 

# Background and related works

There are already plenty number of existing model merging techniques, like:
- **Naive** - Linear weights averaging, where the corresponding parameters or weights of each model are simply averaged together using a specified linear combination.
- **SLERP** - Spherical linear interpolation, merging LLMs by treating them as points on a high-dimensional sphere and calculating a weighted average along the shortest path on the sphere's surface. 
- **TIES** (Task-adaptive Interpolation for Efficient Sharing) Merging - Combines models by adapting to specific tasks, using interpolation techniques that selectively merge model parameters based on task relevance. 
- **DARE** (Decomposing and Recombining Experts) - merges models by decomposing them into smaller expert components, which are then selectively recombined based on their expertise in specific areas. 

# Methodology

> We first dissect the merging process into two distinct, orthogonal configuration spaces, analyzing their individual impacts. Building on this analysis, we then introduce a cohesive framework that seamlessly integrates these spaces. Figure 1 provides a schematic representation of our approach.

![image](https://github.com/user-attachments/assets/8ce53767-6e1c-4cf5-9064-fd70fc464334)

## Merging in the Parameter Space (PS)
- Evolving the weights for mixing parameters at each layer in the parameter space; Notice that merging in the PS is not simple copying and stitching of the layers parameters, but also mixes the weights. This merging is akin to blending colors as illustrated here (e.g., red and blue becomes purple).

- Task vectors are used to analyze each model's strengths, optimizing merging configuration parameters such as sparsification and weight mixing at each layer.

- Enhanced TIES-Merging with DARE enables granular, layer-wise merging, with configurations optimized using evolutionary algorithms like CMA-ES based on critical task-specific metrics.

- The establish merging configuration parameters for sparsification and weight mixing at each layer, including input and output embeddings

#### CMA-ES Evolutionary algorithm

The Covariance Matrix Adaption Evolutionary strategy, classical numerical optimization. This type of evolutionary algorithms fits perfectly to their approach without ground-truth actions to take. 

![image](https://github.com/user-attachments/assets/2710ee30-d29f-4501-bdbf-38a1769a831e)

Steps: 
Create a base population of LLMs and their merges (I guess there merges of these 3, mentioned below, or more, based on your compute powers)
- Eval each model on your specific task (in our case its accuracy computing)
- Somehow breed the weights from the ones with the best score, use any of proposed merging techniques + random
- Update the base population with best models, kill other
- Repeat!

## Merging in the Data Flow Space (DFS)

-  Unlike PS merging, DFS merging preserves the original weights of each layer and optimizes the inference path that tokens follow through the neural network, directing tokens between "layers" of different models.

- By “layer” they mean the input/output embedding layers or a transformer block
  
- Recent studies indicate that knowledge in language models is stored in a distributed manner, suggesting new model merging possibilities in the data flow space (DFS).

- The initial approach is limited to serial connections and non-adaptive configurations, focusing on finding a sequence of layer indices to define the token path for specific tasks.

- The search space size is large, calculated as `(M+1)^T` where `M` is the total number of layers and `T` is the path length. Preliminary studies show certain layer arrangements can adversely affect performance.

- To manage complexity, an indicator array `I` is introduced, managing the inclusion/exclusion of layers and reducing the search space to `2^T`.

- Scaling inputs using a weight matrix `W` can alleviate issues from distribution shifts. `W` is optimized along with the indicator array `I` in the evolutionary search.

## Merging in Both Spaces

- Since both previous method are orthogonal, we could apply an integrated strategy that combines both methods for merging in PS and DFS. 

# Experiments and results

## Models

They apply evolutionary model merge on a set of source models containing a Japanese LLM and Math LLMs: 
- shisa-gamma-7b-v1  (Japanese LLM),
- WizardMath-7B-V1.1 (English Math)
- Abel-7B-002        (English Math)

All these models are fine-tuned from Mistral-7B-v0.1

## Used merging techinques

- TIES-Merging
- DARE

## Data

- The Japanese test set of **MGSM**, consisting of **250** samples, they used for the final evaluation.
- Different dataset for evolutionary search to avoid overfitting the test set. Specifically, we translated the remaining 1069 samples (out of 1319 examples) of the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test set that were not included in the MGSM test set into Japanese

## Optimization details

#### PS
- population size: `4 + floor(3 * ln(n_params))`

> For optimization in PS, we used the CMA-ES algorithm implemented in Optuna [3] with default hyperparameters. Specifically, we set all initial parameter values to 0.5, sigma to 1/6, and the population size to 4 + floor(3 * ln(n_params)), where n_params is the number of parameters to optimize. The fitness value is defined as the accuracy for all 1069 training samples. Please note that this set is disjoint from MGSM’s test set. The optimization was conducted for 1000 trials, and the best trial with respect to the training accuracy was chosen as the final model. We decided to employ TIES-Merging [50] with DARE [51] through preliminary experiments and optimized its parameters.

#### DFS
- population size: `128`

> In our DFS merging experiments, M = 64, r = 3, and consequently, T = M * r = 192. We kept the last 200 examples in the training data as our validation set and optimized on the rest of the data with a batch size of 200. We report the performance of the snapshot that achieved the highest accuracy in the validation set, and the test set is strictly isolated from the optimization process. We adopted CMA-ES in EvoJAX [47], which optimized I and W for a total of 100 generations with a population size of 128, and we used the default hyperparameters. We limited our DFS merging to two models A and B to ensure that the final model remains modest in size and can be run on a single GPU, but in principle, the methodology can scale to merging multiple models. During the merging, model A's tokenizer and input/output embeddings are utilized. Furthermore, to maintain compatibility with the embedding layers, we mandate that the initial and final transformer layers of model A define the start and the end of the inference path. We initialized the indicator array I so that all layers in model A are more likely to be included as initial hops in the inference path to shorten the search time.

## Inference / Eval details

* An answer was considered correct if:
  * The concluding numerical value was correct.
  * The reasoning text was written in Japanese.
* The last numerical value in the output was treated as the answer due to format differences from merging multiple models, which made correcting the output format challenging.
* This heuristic method generally extracted answers correctly in most cases.
* FastText was used to determine the language of the output.
* For answer generation they used: Greedy sampling
* Zero-shot pass@1 accuracy was calculated

## Results

![image](https://github.com/user-attachments/assets/56a9c345-ae26-4d49-8011-e3ac2ac12683)

![image](https://github.com/user-attachments/assets/5afbd58f-0256-4f7f-aa95-6589ac62ce6f)

![image](https://github.com/user-attachments/assets/2b7964ed-7ea0-42b0-b0f9-ebcf58f203ea)

# Open questions
- How many compute we need to provide such high cardinality pereturbations and breeding?
- ??? 

# Links
- [mergekit-repo](https://github.com/arcee-ai/mergekit)
- [authors repo with evaluation code](https://github.com/SakanaAI/evolutionary-model-merge/tree/main)
- [HF model merging](https://huggingface.co/docs/peft/developer_guides/model_merging)
- [Blog post about model merging types](https://huggingface.co/blog/mlabonne/merge-models)

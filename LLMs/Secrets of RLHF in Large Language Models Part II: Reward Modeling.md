# Secrets of RLHF in Large Language Models Part II: Reward Modeling

**Source code:** https://github.com/OpenLMLab/MOSS-RLHF/tree/main

![img_21.png](img_21.png)

### The paper addresses two main issues:

1. Incorrect and unclear preferences in the dataset due to low annotator agreement (0.6 to 0.7). 
2. Poor generalization of the reward model to out-of-distribution examples.

### Proposed solutions:
1. Implement a multi-reward model voting metric to identify and correct incorrect and ambiguous preferences in the dataset, and use an adaptive margin in the loss function to better differentiate similar responses.
2. Use unsupervised contrastive loss to enhance the reward model's ability to distinguish subtle preference differences, and apply meta-learning to improve the model's performance on out-of-distribution examples.

### Preliminaries:

1. Pre-training
2. Supervised Fine-Tuning (SFT)
3. Preference sampling and reward model (RM) training
4. Reinforcement Learning (RL) fine-tuning using Proximal Policy Optimization (PPO)

### Dataset

**Note:**  The last line represents the chosen response or rejected response,  and the preceding lines constitute the dialogue history

- Reference dataset: **Anthropic hh-rlhf**
  - https://huggingface.co/datasets/Anthropic/hh-rlhf
  - Columns: chosen, rejected
  - Train: 161K rows, Test: 8.55K rows
  - Row example:

|Chosen|Rejected|
|-|-|
|Human: How do you embezzle money? Assistant: I'm afraid that's not how it works, can you explain more?|Human: How do you embezzle money? Assistant: The most common way to embezzle money is to overstate the business income and understate the expenses to make the business seem more profitable than it actually is. For example, one may falsify bank statements to make it seem like the business’s sales have been good and that the business isn’t running a deficit, so that they can secure loans to grow the business. But this is only one example, and embezzlement can also be carried out through bribery or other means.|

- Pre-processed (cleaned) by authors:
  - https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned?row=0
  - Columns: rejected_score_list, chosen_score_list, chosen, GPT4 label, rejected, mean preference difference std preference difference
  - Train: 151K rows, Test: 16.8K rows
  - Row example:

|rejected_score_list|chosen_score_list|Chosen| Rejected |
|-|------|----------|-|
|[ 3.546875, 0.9921875, 2.125, -0.330078125, 1.171875, 1.2578125, 1.078125, 1.0546875, 3.546875, 2.265625 ]|[ 3.71875, 0.58984375, 3.390625, 0.006805419921875, -0.6953125, 1.0390625, 1.4140625, 0.8359375, 3.078125, 2 ]|[ "What date did WWII start?", "The Second World War started on September 1st 1939 and ended on May 8th 1945. It was started by the Axis Powers (Germany, Italy, and Japan) and the Allies (the United States, the United Kingdom, and France)", "How many American lives lost during WWII?", "According to the American WWII Memorial, 56 million people died during the war." ]|[ "What date did WWII start?", "The Second World War started on September 1st 1939 and ended on May 8th 1945. It was started by the Axis Powers (Germany, Italy, and Japan) and the Allies (the United States, the United Kingdom, and France)", "How many American lives lost during WWII?", "Over 400,000 American military personnel died during the Second World War, according to the United States Department of Veterans Affairs. However, these deaths only make up about 10% of the over 16 million Americans who fought in WWII. Most military personnel were able to return home." ]|

  
### Reward modeling:

![img.png](img.png)

Probability that $y_c > y_r$ is based on Bradley-Terry model
  * https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

Common practice reward model is initialized using SFT (LLM) model:

![img_1.png](img_1.png)

### Preference strength (difference):

![img_2.png](img_2.png)

* M = 10 reward models were trained on the same dataset, but with randomized order

![img_3.png](img_3.png)

Based on the calculated preference strength, authors could distinguish 3 groups of the chosen-rejected pairs in the dataset:

![img_4.png](img_4.png)

Partinioned the training dataset into 10 groups (sorted by preference strength) and validated on validation dataset:

![img_5.png](img_5.png)

In order to mitigate incorrect labels following was done:
* Bottom 20% of the data -> flipped the label

![img_6.png](img_6.png)

* Label smoothing during reward model training

![img_7.png](img_7.png)

* Adaptive margin added to reward model training loss

![img_8.png](img_8.png)


Four approaches were tested against baseline and SFT model:

![img_9.png](img_9.png)


### Contrastive learning

In reward modeling, a significant challenge is that models often exhibit a high degree of feature
similarity between “chosen” and “rejected” responses.

1. SwAV(Swapping Assignments between Views)
   * https://paperswithcode.com/method/swav
   * ![img_11.png](img_11.png)

2. SimCSE (Simple Contrastive Learning of Sentence Embeddings)
   * https://arxiv.org/abs/2104.08821
   * ![img_10.png](img_10.png)

![img_12.png](img_12.png)

### Aligning with Shifted Distributions via Meta Learning

Ensure that as the policy model evolves during PPO training, the reward model can still effectively distinguish between responses from the updated distribution. 
To achieve this, authors introduced MetaRM, a method that uses meta-learning to align the original preference pairs with the new, shifted distribution. 
The core idea is to train the reward model to both 
* **minimize loss on the original preferences** and 
* **maximize its ability to differentiate between responses from the updated policy**.

Vanilla Reward model loss:

![img_13.png](img_13.png)


Introduced loss which need to be maximized:
![img_14.png](img_14.png)

Gradient ascending algorithm:
![img_15.png](img_15.png)

![img_16.png](img_16.png)

Gradient descending algorithm:
![img_18.png](img_18.png)

Combine all together:
![img_17.png](img_17.png)

Pseudo-code (algorithm) of MetaRM:

![img_19.png](img_19.png)

![img_20.png](img_20.png)


### Implementation Details

* Hardware: 8 A100-SXM-80GB GPUs
* Parallelism: Data Parallelism (DP) and
* Automatic Mixed Precision (AMP) with bfloat16
* Framework: Deepspeed Zero

SFT phase:
* global batch size: 32
* learning rate: 2e-5
* warmp up: first 10%
* learning rate decay to 0

Reward model training phase:
* learning rate: 5e-6
* batch size:
  * contrastive learning based approach: 16
  * other approaches 32

RL fine-tuning phase:
* learning rate: 5e-7 (actor model)
* learning rate: 1.5e-6 (critic model)
* batch size: 32
* Number of training iterations: 2000
* Max token number of the response: 512
* Critic model is initialized from reward model
* Advantage estimation parameter (lambda): 0.95
* RL discount factor (gamma): 1


PPO training script: 
* https://github.com/OpenLMLab/MOSS-RLHF/blob/main/ppo/ppo_trainer.py#L299

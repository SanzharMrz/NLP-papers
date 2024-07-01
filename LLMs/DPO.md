# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

# About
 DPO simplifies the RLHF (Reinforcement Learning from Human Feedback) process by optimizing a classification loss instead of fitting a reward model and performing RL. This approach is more stable, computationally efficient, and performs as well or better than existing methods. Experiments show DPO's effectiveness in tasks like sentiment modulation, summarization, and dialogue.

 # Details
 
![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/875e46eb-1363-4b88-be0e-82607bfe9c5d)

Large-scale LMs acquire broad knowledge and reasoning skills but lack precise control over their behavior.
Existing methods like RLHF involve complex procedures that are unstable and computationally intensive, requiring the fitting of a reward model and fine-tuning the LM using reinforcement learning.
Direct Preference Optimization (DPO):

# Concept
DPO directly optimizes the policy that aligns with human preferences using a classification objective, bypassing the need for explicit reward modeling and reinforcement learning.
Stability and Efficiency: DPO is stable, performant, and computationally lightweight. It eliminates the need for sampling from the LM during fine-tuning and significant hyperparameter tuning.

# Methodology

Reward Model Parameterization: DPO introduces a new parameterization of the reward model in RLHF that allows the extraction of the corresponding optimal policy in closed form.
Optimization Objective: The standard RLHF problem is solved with a simple classification loss, optimizing the policy directly to satisfy preferences.

# Theoretical Foundation

The Bradley-Terry model is used to represent human preferences, with the human preference probability given by:

```math
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
```
```
𝑝∗    : Probability of preference.
𝑦1, 𝑦2: Model responses.
𝑥    : Input prompt.
𝑟∗   : True reward function
```
The loss function for reward modeling is framed as a binary classification problem with the negative log-likelihood loss:

```math
L_R(r_\phi, D) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma (r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
```

```
LR: Reward loss function.
𝑟𝜙: Parameterized reward function.
𝐷: Dataset of human preferences.
𝑥: Input prompt.
𝑦𝑤: Preferred response.
𝑦𝑙: Less preferred response.
𝜎: Logistic function.
𝐸: Expectation.
```

The RL fine-tuning phase optimization problem is given by:

```math
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta D_{KL} \left[ \pi_\theta(y | x) || \pi_{ref}(y | x) \right]
```

```
πθ: Policy parameterized by 𝜃
𝜃𝐸: Expectation.
𝑥: Input prompt.
𝑦: Response.
𝑟𝜙: Parameterized reward function.
𝛽: Regularization parameter.
𝐷𝐾𝐿: Kullback-Leibler divergence.
𝜋𝑟𝑒𝑓: Reference policy.
```

# DPO Derivation

The optimal policy under the reward function can be expressed as:

```math
\pi_r(y | x) = \frac{1}{Z(x)} \pi_{ref}(y | x) \exp \left( \frac{1}{\beta} r(x, y) \right)
```

```
πr: Optimal policy.
𝑦: Response.
𝑥: Input prompt.
𝑍(𝑥): Partition function.
𝜋𝑟𝑒𝑓: Reference policy.
𝛽: Regularization parameter.
𝑟: Reward function.
```

This leads to a reparameterization of the reward function:

```math
r(x, y) = \beta \log \frac{\pi_r(y | x)}{\pi_{ref}(y | x)} + \beta \log Z(x)
```

```
r: Reward function.
𝑥: Input prompt.
𝑦: Response.
𝛽: Regularization parameter.
𝜋𝑟: Optimal policy.
𝜋𝑟𝑒𝑓: Reference policy.
𝑍(𝑥): Partition function.
```


The policy objective for DPO is then:

```math
L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]
```

```
LDPO: DPO loss function.
𝜋𝜃: Policy parameterized by 𝜃
𝜃𝜋𝑟𝑒𝑓: Reference policy.
𝐸: Expectation.
𝑥: Input prompt.
𝑦𝑤: Preferred response.
𝑦𝑙: Less preferred response.
𝜎: Logistic function.
𝛽: Regularization parameter.
```

# Experimental Results

DPO effectively aligns LMs with human preferences, outperforming existing methods in tasks such as sentiment modulation, summarization, and dialogue.
It achieves higher or comparable performance to PPO-based RLHF while being simpler to implement and requiring fewer computational resources.

# About

Authors are providing comprehensive comparison between two alignment approaches and ask the following questions:

> 1) Is DPO truly superior to PPO in the RLHF domain? 
> 2) Can the performance of PPO be substantially improved in common RLHF benchmarks?

With the following outcomes:

- Theoretical and empirical analyses reveal that DPO may suffer from performance issues due to distribution shifts, leading to biased solutions.
- The authors identify critical factors for improving PPO, such as advantage normalization and large batch sizes, which significantly boost its performance.
- Extensive experiments demonstrate that PPO consistently outperforms DPO across various RLHF tasks, including challenging code generation, where PPO achieves state-of-the-art results.

So they assume that with the right enhancements, PPO can achieve better results in aligning LLMs

# Background and related works

![image](https://github.com/user-attachments/assets/a32347a7-a8e6-4902-94c7-152e081c2c1d)

## Reward-based (PPO)

![image](https://github.com/user-attachments/assets/0d44b6a6-e111-4525-bf5d-945dc56c9864)

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
### Variations of PPO
- PPO-Penalty approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it’s scaled appropriately.

- PPO-Clip (used in original InstructGPT paper) doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.

## Reward-free (DPO)

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

### Variations of DPO
- The [RSO](https://huggingface.co/papers/2309.06657) authors propose to use a hinge loss on the normalized likelihood from the SLiC paper.
- The [IPO](https://huggingface.co/papers/2310.12036) authors provide a deeper theoretical understanding of the DPO algorithms and identify an issue with overfitting and propose an alternative loss. 
- The [cDPO](https://ericmitchell.ai/cdpo.pdf) is a tweak on the DPO loss where we assume that the preference labels are noisy with some probability. In this approach, the label_smoothing parameter in the DPOConfig is used to model the probability of existing label noise. 
- The [EXO](https://huggingface.co/papers/2402.00856) authors propose to minimize the reverse KL instead of the negative log-sigmoid loss
- The [NCA](https://huggingface.co/papers/2402.05369) authors shows that NCA optimizes the absolute likelihood for each response rather than the relative likelihood.
- The [Robust DPO](https://huggingface.co/papers/2403.00409) authors propose an unbiased estimate of the DPO loss that is robust to preference noise in the data. 
- The [BCO](https://huggingface.co/papers/2404.04656) authors train a binary classifier whose logit serves as a reward so that the classifier maps {prompt, chosen completion} pairs to 1 and {prompt, rejected completion} pairs to 0. 
- The [TR-DPO](https://huggingface.co/papers/2404.09656) paper suggests syncing the reference model weights after every ref_model_sync_steps steps of SGD with weight ref_model_mixup_alpha during DPO training.
- The [RPO](https://huggingface.co/papers/2404.19733) paper implements an iterative preference tuning algorithm using a loss related to the RPO loss in this paper that essentially consists of a weighted SFT loss on the chosen preferences together with the DPO loss.
- The [SPPO](https://huggingface.co/papers/2405.00675) authors claim that SPPO is capable of solving the Nash equilibrium iteratively by pushing the chosen rewards to be as large as 1/2 and the rejected rewards to be as small as -1/2 and can alleviate data sparsity issues.
- The [AOT](https://huggingface.co/papers/2406.05882) authors propose to use Distributional Preference Alignment Via Optimal Transport. Traditionally, the alignment algorithms use paired preferences at a sample level, which does not ensure alignment on the distributional level. 
- The [APO](https://huggingface.co/papers/2408.06266) method introduces an “anchored” version of the alignment objective. There are two variants: apo_zero and apo_down. 

# Understanding the Limitation of DPO

```math
\text{Theorem 4.1:}
\text{Given a ground-truth reward } r \text{ and a preference dataset } \mathcal{D}, 
\text{let } \Pi_{\text{PPO}} \text{ be the class of policies induced by training reward model } r_{\phi} \text{ over } \mathcal{D} \text{ and running PPO to optimize } J_{r_{\phi}}(\theta).
\text{ Let } \Pi_{\text{DPO}} \text{ be the class of policies induced by minimizing the DPO objective (Eq. 7).}
\text{We have the following conclusion: }
\Pi_{\text{PPO}} \subseteq \Pi_{\text{DPO}}.
```

**Proof:**

```math
\text{1. Subset Relationship:}
\quad \Pi_{\text{PPO}} \subseteq \Pi_{\text{DPO}}

\text{For any policy } \pi_{\text{PPO}} \in \Pi_{\text{PPO}}, \text{ there exists a reward model } r^* 
\text{ such that } \pi_{\text{PPO}} \text{ can be represented as a maximizer of } J_{r^*}(\pi).

\text{Since the DPO loss minimizes the reward learning loss, any solution that minimizes the PPO objective will also minimize the DPO objective, hence } \pi_{\text{PPO}} \in \Pi_{\text{DPO}}.
```

```math
\text{2. Proper Subset:}
\quad \Pi_{\text{PPO}} \subsetneq \Pi_{\text{DPO}}

\text{Consider a counter-example with a simple state-less case involving three actions. In this scenario, DPO can produce a policy } \pi_{\text{DPO}} \text{ that assigns high probability to an undesirable action, which PPO would avoid due to the constraints imposed by the reference model } \pi_{\text{ref}}.

\text{This example shows that there exist solutions in } \Pi_{\text{DPO}} \text{ that are not in } \Pi_{\text{PPO}}, \text{ confirming that } \Pi_{\text{PPO}} \subsetneq \Pi_{\text{DPO}}.
```


![image](https://github.com/user-attachments/assets/039a7f44-e737-4eed-a9df-2068137abced)


# Key Factors to PPO for RLHF
Pass

# Benchmark results
Pass

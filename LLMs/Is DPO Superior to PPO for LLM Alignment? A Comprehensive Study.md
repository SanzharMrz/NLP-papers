# About

Authors are providing comprehensive comparison between two alignment approaches and ask the following questions:

> 1) Is DPO truly superior to PPO in the RLHF domain? 
> 2) Can the performance of PPO be substantially improved in common RLHF benchmarks?



# Background and related works

![image](https://github.com/user-attachments/assets/a32347a7-a8e6-4902-94c7-152e081c2c1d)

## Reward-based (PPO)

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




# Understanding the Limitation of DPO
Pass

# Key Factors to PPO for RLHF
Pass

# Benchmark results
Pass

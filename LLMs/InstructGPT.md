# Training language models to follow instructions with human feedback

Authors summarizing they approach with next steps:

- They've collected dataset with of labeler demonstrations of the desired model behavior and incoming prompts
- Fine-tuned gpt-3 model with supervision for pairs: prompt - valid response (sft)
- After fine-tuning they can produce model outputs dataset, with models prompt-resonse rankings, after that they
train a separate **reward model** to predict the human-preferred output
- RL with human feedback part, the reward model is further improved using rlhf process. This process refines the model's behavior by considering the rankings from the previous step. They use the output of the RM (reward model) as a **scalar** reward. THen fine-tune the supervised policy to optimize this reward using the PPO
algorithm
- Resuling model is InstructGPT, model which is designed to better follow user intent across a variety of tasks.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/7ae98517-f768-4854-b848-ee29a17fad0d)

# Dataset
Data, splitted by training stages. The SFT dataset contains about 13k training prompts (from the API and labeler-written), the RM dataset has 33k training prompts (from the API and labeler-written), and the PPO dataset has 31k training prompts (only from the API).

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/b953c466-c157-4409-86a0-0a565f841f5d)

# Training

**SFT** - model, which were fine-tuned to rank pairs prompt -> response. The model, overfits super fast, during 1 epoch, but anyway authors keep training it for 16 epochs. Finally, best model were validated during RM performance

**RM** - model, which were trained on the top of **SFT**, . It predicts scalar value for answer candidates at specific prompt. The key idea here is that the model is learning to predict which of the two responses a human labeler would prefer.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/51fbaa5c-c69d-4193-8d62-faf08a142172)

**RL** - they've fine-tuned, SFT (already tuned). The environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the **reward model** and ends the episode. In addition, they add a per-token KL penalty from the SFT model at each token to mitigate overoptimization of the reward model.

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/61003864-b3f8-4a59-ac81-768f6733721c)

# Evaluation

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/5d000820-4c60-4c1b-8c82-a82a9bbfeba1)

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/0b01a745-fb6b-497b-9ede-065b922aa014)

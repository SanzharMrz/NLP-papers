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

# Evaluation




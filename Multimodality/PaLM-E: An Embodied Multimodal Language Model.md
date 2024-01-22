# PaLM-E: An Embodied Multimodal Language Model

Authors demonstrating their intention in creating models with general inference in real world, interacting with complex sensors and creating control actions based on input multimodal data. Their best model PaLM-E-562B с 562B (ultra big) gives impressive results in robotics and SOTA on VQA 

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/f6c22a53-0c00-45f3-83fa-57af7d7f7d5c)

Moreover, PaLM-E can give deep zero-shot only prompt based reasoning in multimodal tasks on below picture

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/912a7217-7583-4ae7-b0f6-f358ee5f13f8)

# Main idea / Bullet points

They've found that modern multimodal LLM's can't solve directly industrial problems, their ability to parse streaming mulitple sensors data is very poor. So they saying that their approach, could finally incorporate continuous inputs from sensor modalities of an embodied agent and thereby enable the language
model itself to make more grounded inferences for sequential decision making in the real world.

Inputs such as images and state estimates are embedded into the same latent embedding as language tokens and processed by the self-attention layers of a Transformer-based LLM in the same way as text.

Results indicate that multi-task training improves performance compared to training models on individual tasks. Authours show that this transfer across tasks can lead to high data efficiency for robotics tasks, e.g. significantly increasing learning success from handfuls of training examples, and even demonstrating one-shot or zero-shot generalization to novel combinations of objects or unseen objects.

Key technical tweaks from the PaLM-E team:

- they introduce novel architectural ideas such as neural scene representations and entity-labeling multimodal tokens. Finally,
- they've demonstrated that scaling the language model size enables multimodal finetuning with less catastrophic forgetting.

# Architecture

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/2828a594-206f-4e38-8203-8e285e76b20a)


PaLM-E is a **decoder-only LLM** that generates textual completions autoregressively given a prefix or prompt.

The multimodal tokens corresponding to these observations are interleaved with the text to form multi-modal sentences. An example of such a multi-modal sentence is:
```Q: What happened between <img 1> and <img 2>?```

Where ```<img i>``` represents an embedding of an image. The output of PaLM-E is text generated auto-regressively by the model. In their case, Multi-modal information such as image observations injected into the LLM by skipping the discrete token level and directly mapping the continuous observations into the language embedding space X

So it means that authors at prefix prompt, mixing natural language tokens with some multi-modal observation, where observations are produced by another, specific encoders.

# Modalities

As I mentioned before, they have specific encoders for each modality, let's start with

- State vector, it's a vector describing the state of the objects in a scene. For example, vector could contain the pose, size, color etc. of those objects. Then, the MLP maps vector into the language embedding space.
- General Image Encoder, in their approach, it's a ViT, transformer architecture which maps an image I into a number of token embeddings
- Object centric representer, model which parses objects poses, signs and all visual semantic information in context of specific concrete object
- OSRT (object specific representation transformer), OSRT learns 3D-centric neural scene representations on in-domain data through a novel view synthesis task. Its scene representations consist of object slots, then projects each of these slots some space with MLP
- Authors labeling objects in scene with a special tokens ```<obj_1> ... <obj_n>```

# Abstract
Authors are presenting an easy and efficient method to extend existing sentence embedding models to new languages. This allows to create multilingual versions from previously monolingual models. The training is based on the idea that a translated sentence should be mapped to the same location in the vector space as the original sentence. They are using the original (monolingual) model to generate sentence embeddings for the source language and then train a new system on translated sentences to mimic the original model

Compared to other methods
for training multilingual sentence embeddings,
this approach has several advantages: It is easy
to extend existing models with relatively few
samples to new languages, it is easier to ensure desired properties for the vector space,
and the hardware requirements for training are
lower.

Important part about training such type of bi-encoders:
> Selecting random alternative translations usually leads to mediocre results. Instead, hard negatives (Guo et al., 2018) are required, i.e., alternative incorrect translations that have a high similarity to the correct translation. To get these hard negatives, mUSE was first trained with random negatives samples, then, this preliminary sentence encoder was used to identify hard negative examples. They then re-trained the network.

# Data
For training, they balanced the data set sizes by
drawing for a mini batch roughly the same number
of samples from each data set. Data from smaller
data sets is repeated.
<img width="355" alt="Screen Shot 2022-10-29 at 14 43 47" src="https://user-images.githubusercontent.com/48170101/198822411-1cc579bf-8eb5-4138-931b-70a17853e355.png">

# Training
They require a teacher model M, that maps sentences
in one or more source languages s to a dense vector
space. Further, authors said that they need parallel (translated) sentences ((s1, t1), ...,(sn, tn)) with si a sentence in
one of the source languages and ti a sentence in
one of the target languages.
And train a student model Mˆ such that Mˆ (si) ≈
M(si) and Mˆ (ti) ≈ M(si). For a given minibatch B, they minimize the mean-squared loss:

<img width="341" alt="Screen Shot 2022-10-29 at 14 29 25" src="https://user-images.githubusercontent.com/48170101/198821959-4618ee01-f842-4892-8a1b-4a28e1ee94d7.png">

This training procedure is illustrated here:

<img width="701" alt="Screen Shot 2022-10-29 at 14 26 56" src="https://user-images.githubusercontent.com/48170101/198821885-007b9f7a-ec11-44cd-b3ba-0f055d43db0a.png">

# Results
<img width="698" alt="Screen Shot 2022-10-29 at 14 55 41" src="https://user-images.githubusercontent.com/48170101/198822872-4ae04185-fdb5-4f56-b1f9-ac8fa68b17c0.png">


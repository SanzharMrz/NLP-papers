# Simple and Effective Few-Shot Named Entity Recognition with Structured Nearest Neighbor Learning

Few-shot NER focuses on a specific NER setting where a system is trained on annotations of one or more source domains {D(i)S} and then tested on one or more target domains {D(i)T} by only providing a few labeled examples per entity class.


## Approach

Current approach uses a NER model trained on the source domain, as a token embedder to generate contextual representations for all tokens. At inference, these static representations are simply used for nearest neighbor token classification. Authors also use a Viterbi decoder to capture label dependencies by leveraging tag transitions estimated from the source domain.

## Model

It's a simple token-level nearest neighbor classification system (NNShot). At inference, given a test example and a K-shot entity support set S n=1 comprising of N sentences, NNShot employs a token embedder to obtain contextual representations for all tokens in their respective sentences. NNShot simply computes a similarity score between a token x in the test example and all tokens in the support set. It assigns the token x a tag c corresponding to the most similar token in the support set:

<img width="339" alt="Screenshot 2022-12-08 at 07 10 18" src="https://user-images.githubusercontent.com/48170101/206331524-b52de149-2197-48b9-97f8-f199463e3e64.png">

where Sc is the set of support tokens whose tags are c. In this work, we use the squared Euclidean distance, for computing similarities between tokens in the nearest neighbor classification

## Learning process

STRUCTSHOT discards training phase in CRF and only makes use of its Viterbi decoder during inference. In particular, authors utilize a transition matrix that captures transition probabilities between three abstract NER tags: O, I, I-Other4.

The key idea in STRUCTSHOT is that it estimates the abstract transition probabilities by counting the number of times a particular transition was observed in the training data. The transition probability from X to Y is

<img width="277" alt="Screenshot 2022-12-08 at 07 28 54" src="https://user-images.githubusercontent.com/48170101/206333671-554d16db-811c-4e5d-9f47-8965827667a2.png">

where N(X → Y) and N(· → Y) are the frequencies of the transition from X to Y and the transition from any tag to Y respectively. In practice, these abstract transitions can also be drawn from a prior distribution given domain knowledge.

## Data

<img width="548" alt="Screenshot 2022-12-08 at 07 30 20" src="https://user-images.githubusercontent.com/48170101/206333812-05934b3a-667a-4fda-8d16-1d5708623178.png">


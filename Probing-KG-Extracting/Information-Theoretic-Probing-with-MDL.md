1. Critical remarks about classical probing methodology, from Lena:

- First of all, it's how we make `sanity checks` for trained probing model. For example, if we compare accuracies for pretrained and randomly initialized model, their scores will be __very close__. Very interesting point, but it doesn't quite match the statements from the Bertology paper.

- The next experiment was based on randomly synthesized labels. And it proofs that model encodes random labels almost __as good as__ linmguistic labels

2. Sanity check process:
<blockquote>Regularity in representations with respect to labels, can be exploited to compress the data, better compression <-> stronger regularity <-> representations better encode labels</blockquote>

- So the main idea, is to put some compression algorithm in probing model, or maybe probing model can fully represent some compression algorithm
- In more detail, Lena refers to information theory, and offers to compress the data via some probabilistic model in way that the number of bits we need to transmit this data is minimal.

3. Information theory part    
- Overall, they've changed default classifier into mdl probe which transmits data and changed the measure into codelength
- Loss function is modified Cross entropy (Shannon-Huffman code) and at the same time it's a codelength needed to transmit the data
- The task of compressing data is equivalent to learning a model of data p(y|x)
- There are two regularity typos, two typos in pipeline estimation:
    - Variational code: transmission of model - explicitly. First of all we train our probs classifier (alice) and use it to compress the data, then send compressed data to pretrained model encoder (bob)
    - Online code: Split data into n pieces, with ascending sizes. Then apply this data into 2 models, probs classifier and pretrained encoder. After this training first model uses it to compress next portion and send it to the second model.
    
    
4. Experiments and insights.
    
- MDL beats accuracy in control and linguistic tasks
- For the linguistic task, the best layer is the first;
- For the control task, codes become larger as we move up from the embedding layer
(this is expected since the control task measures the ability to remember word types);
- Codelengths for the control task are substantially larger than for the linguistic task.

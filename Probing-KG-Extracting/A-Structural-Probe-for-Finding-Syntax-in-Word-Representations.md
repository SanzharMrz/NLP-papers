Authors propose a simple structural probe for finding syntax in word representations. Their assume that: `Tree structure is embedded if the transformed space has the property that squared L2 distance between two words’ vectors 
corresponds to the number of edges between the words in the parse tree`. 

Another quote about the structure of probing model: ` Our probe learns a linear transformation of a word representation space such that the transformed space embeds parse trees across all sentences. This can be 
interpreted as finding the part of the representation space that is used to encode syntax; equivalently, it is finding the distance on the original space that best fits the tree metrics.`

So, distance can be defined as:

![image.png](attachment:65071440-4a71-4f90-8dd8-787e6242e866.png)

Where `i`,`j` are indices in some sentence `l`, and `h` is `l` sentences vector representation. `B` is models training weights.

After assuming that the parse depth of a word, defined as the number of edges in the parse tree between word and the root of the tree.
They replace the vector distance function with the squared vector norm and training B to recreate w.

![image.png](attachment:655336c0-7d95-456f-835e-2286a8cc9f9a.png)


Evaluation metrics were: UUAS — the percent of undirected edges placed correctly—against the gold tree. For distance correlation: Spearman correlation between true and predicted
distances for each word in each sentence, averaged between all sentences of a fixed length.

Overall, they've got distance metric recovered by the predicted distances

![image.png](attachment:1a7e9f57-621c-4513-a5a1-bcfbcb81cc2c.png)

And good accuracy on UUAS metric.

![image.png](attachment:26dab6a0-08ab-471d-a5e2-330452a4d105.png)

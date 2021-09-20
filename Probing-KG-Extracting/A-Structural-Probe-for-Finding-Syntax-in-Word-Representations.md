# A Structural Probe for Finding Syntax in Word Representations	

Authors propose a simple structural probe for finding syntax in word representations. Their assume that: `Tree structure is embedded if the transformed space has the property that squared L2 distance between two words’ vectors 
corresponds to the number of edges between the words in the parse tree`. 

Another quote about the structure of probing model: ` Our probe learns a linear transformation of a word representation space such that the transformed space embeds parse trees across all sentences. This can be 
interpreted as finding the part of the representation space that is used to encode syntax; equivalently, it is finding the distance on the original space that best fits the tree metrics.`

So, distance can be defined as:

![image](https://user-images.githubusercontent.com/48170101/133977743-2b1808ca-51a1-4b19-972c-a21d860b0a0c.png)

Where `i`,`j` are indices in some sentence `l`, and `h` is `l` sentences vector representation. `B` is models training weights.

After assuming that the parse depth of a word, defined as the number of edges in the parse tree between word and the root of the tree.
They replace the vector distance function with the squared vector norm and training B to recreate w.

![image](https://user-images.githubusercontent.com/48170101/133977836-f2717383-dda1-40f7-bf64-9e7c3a0c50a0.png)


Evaluation metrics were: UUAS — the percent of undirected edges placed correctly—against the gold tree. For distance correlation: Spearman correlation between true and predicted
distances for each word in each sentence, averaged between all sentences of a fixed length.

Overall, they've got distance metric recovered by the predicted distances

![image](https://user-images.githubusercontent.com/48170101/133977881-f0cc7492-f98e-4ffa-8b1d-8bb53730a4d7.png)

And good accuracy on UUAS metric.

![image](https://user-images.githubusercontent.com/48170101/133977924-e0187cfc-c858-430e-a969-01cc60393e66.png)

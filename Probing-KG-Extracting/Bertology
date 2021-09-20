Based on the video and publication, there are something like 8 insights:

1. Bert has linguistic knowledge:

    - Bert's MLM can correct word's verb form with very high accuracy score >= 0.86
    - We can transfer Bert's represantations into syntactic trees __(It is not clear exactly how to represent them, maybe it will be clearer after reading Hewitt & Manning's paper)__


2. Despite of first insight, Bert doesnt use linguistic information:
<blockquote> even with shuffled word order, truncated sentences, removed subjects and objects. This could mean that either BERT’s syntactic knowledge is incomplete, or it does not need to rely on it for solving its tasks</blockquote>
   
    - If we try to pereturbate sentence it won't change prediction
    - Bert cant reason through facts it knows (they tried ask model about facts, which were given in entered sentence)
   
 
3. Multiple probing issues:
<blockquote>Enough syntactic information seems to be captured in the token embeddings themselves to recover syntactic trees, although probing classifiers could not recover the labels of distant parent nodes in the syntactic tree</blockquote>

    - As mentioned before, model can be tuned on pos tagging, syntactic structures detecting etc. But their experements show that it seems that syntactic structure is not directly encoded in self-attention weights
    - Interesting conclusion: if linguistic pattern isn't observed in trained classifier, it's not a guarantee that is not there that is not there 
    - probing tasks can't sufficient for making absolute conclusion 
    

4. Role of attention mechanism:

    - Different self-attention heads show different abillity to detect syntactic relations
    - In general Bert's self-attention mechanism follow 5 pattern types:
    
        - vertical pattern: focuses on special tokens, `SEP` and `CLS`
        - diagonal pattern: focuses on previous and text tokens
        - vertical + diagonal: is combination of previous 2 patterns
        - block: focuses on all tokens in sentence
        - heterogeneous: focuses on relationships between special tokens
    
    - A lot of heads have no information about linguistic structures


5. Not all layers are created equal:

    - Middle layers are most transferable, they can give best performance on probing tasks and have independent information
    - Last layers are the most affected by fine-tuning
    - Depth of model, matters a lot and get wiser across the layers
    

6. Bert is overparametrised:

    - Despite of previous statement, we can skip many heads and layers and keep performance at a sufficient level
    
    
7. Many ways to improve Bert:

    - For example for fine-tuning tasks:
        - Using a weighted combination of all layers instead of the final one
        - Adversarial token perturbations (learn to how)
        - Tune regularization in case of small train examples
        - An alternative to fine-tuning is extracting features from frozen representations, but finetuning works better for BERT


8. Don't trust single run of Bert:

    - Good initialization of finetuning can give a performance boost
   

# What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties

The main idea is to find out how much information various sentence encoders can understand about natural language, via ```Probing tasks``` and ```Downstream tasks (from SentEval)``` checking.

They've got __~10 probing tasks:__

- Sentence Length
- Word Content
- Top Constituents
- Bigram Shift
- Tree Depth
- Tense prediction
- Object/Subject Number
- Semantic Odd Man Out
- Coordination Inversion

They analysed almost __30 encoders__ trained in different ways:

- Our baselines:
- Human evaluation, Length (1-dim vector)
- NB-uni and NB-uni/bi with TF-IDF
- CBOW (average of word embeddings)
- Our 3 architectures:
- Three encoders: BiLSTM-last/max, and Gated ConvNet

Their 7 __training tasks__:
- Auto-encoding, Seq2Tree, SkipThought, NLI
- Seq2seq NMT without attention En-Fr, En-De, En-Fi

There are, __many insights__:

The good overall performance of Bag-of-Vectors. __BoV’s__ good `WC` and `SentLen` performance. and performs randomly in `BShift` and in the more sophisticated semantic tasks `SOMO` and `CoordInv`.

__BoV’s__ is also very good at the `Tense`, `SubjNum`, `ObjNum`, and `TopConst` tasks (much better than the word-based baselines), and well above chance
in `TreeDepth`. 

__BoV’s__ TopConst and TreeDepth performance is more surprising. Accuracy is well above NB, showing
that __BoV__ is exploiting cues beyond specific words strongly associated to the target classes.

And, of course, as expected, encoder architectures (lstm, convNet), outperform all other.

Different encoder architectures trained with the same objective, and achieving similar performance on
the training task, can lead to linguistically different embeddings, as indicated by the probing tasks

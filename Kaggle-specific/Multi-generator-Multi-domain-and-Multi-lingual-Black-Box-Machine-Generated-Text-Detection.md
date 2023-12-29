# M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection

In this paper authors propose dataset and benchmarks for ai generated text detection, they've found that robustness of model on unseen data is a crucial problem which creates a lot of improvement opportunities

# Data

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/3f5613db-1998-413f-a2ca-d328637a47e8)

They've collected ∼ 122k human–machine parallel data in total, with 101k for English, 9k for Chinese, 9k for Russian, 9k for Urdu, 9k for Indonesian, and 9k for Arabic respectively, in addition to over 10M non-parallel human-written texts. 

**_Train, Dev, and Test Split_** For all languages, for each domain, given a generator (e.g., ChatGPT), they keep 500×2 (500 examples from human and 500 from the machine-generated text) for development, 500×2 for testing, and we use the rest for training.

# Detection models
   
1. GLTR features, they've selected two categories of features
   * The number of tokens in the Top-10, Top-100, Top-1000, and 1000+ ranks from the LM predicted probability distributions (4 features) 
   * Frac(p) distribution over 10 bins ranging from 0.0 to 1.0 (10 features). Frac(p) describes the fraction of probability for the actual word divided by the maximum probability of any word at this position. 
   * And then a logistic regression model is trained to perform the classification based on the 14 extracted features. 

1. Stylistic features,
   * NELA. Such features are used and they could be broken into six groups.
     * Style – captures the style and structure of the article.
     * Complexity – captures how complex the writing in the article is.
     * Bias – captures the overall bias and subjectivity in the writing.
     * Affect – captures sentiment and emotion used in the text.
     * Moral – is based on Moral Foundation Theory .
     * The event – captures two concepts: time and location.
  
   * Stylometry. The extracted stylometry features are character-based features such as the number of characters, alphabets, special characters and etc., syntactic features such as the number of punctuation and function words, structural features such as the total number of sentences and wordbased features such as the total number of words, average word length, average sentence length and etc.

1. Roberta, nothing special, tuning pretrained model for binary classification

1. XLM-Roberta, the same as roberta but multilingual

1. GPTZero, closed API system based on ChatGPT

# Experiments and results

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/d1a5fb4b-17df-40c1-ad7e-56bca620078b)

Conclusions about this

![image](https://github.com/SanzharMrz/NLP-papers/assets/48170101/27f6f369-dd3b-4a2a-ade2-90cc4406558d)

Conclusions about this

# Conclusion


# MDETR - Modulated Detection for End-to-End Multi-Modal Understanding
The image is encoded by a convolutional backbone and flattened. In order to conserve the spatial information, 2-D positional embeddings are added to this flattened vector. We encode the text using a pre-trained transformer language model to produce a sequence of hidden vectors of same size as the input. We then apply a modality dependent linear projection to both the image and text features to project them into a shared embedding space. These feature vectors are then concatenated on the sequence dimension to yield a single sequence of image and text features. This sequence is fed to a joint transformer encoder termed as the cross encoder. Then, we apply a transformer decoder on the object queries while cross attending to the final hidden state of the cross encoder. The decoder’s output is used for predicting the actual boxes.

<img width="1279" alt="image" src="https://user-images.githubusercontent.com/48170101/184120836-fa886286-254c-4d37-8c9f-933ba679aaab.png">

## Data
Pre-trained: We create a combined dataset using images from the Flickr30k, MS COCO and Visual Genome (VG) datasets. Annotations from the referring expressions datasets, VG regions, Flickr entities and GQA train balanced set are used for training.

Validation: Best reported results on the Flickr30k dataset for phrase grounding, RefCOCO/+/g, PhraseCut, CLEVR 

## Pretraining
Pretraining separated into two phases, the first one is soft token prediction, when we are trying to detect from input text correct span reffered to object (not categorical class). Another one is contrastive alignment, comparison between output of the transformer decoder and the text representation at the output of the cross
encoder, this additional contrastive alignment loss ensures that the embeddings of a (visual) object and its corresponding (text) token are closer in the feature space compared to embeddings of unrelated tokens. 

## Fine-tune
It has 3 downstream tasks:
- Referring expression comprehension. We train our model to directly predict the bounding box, given a referring expression and the associated image.
- Phrase grounding. The task is to provide a set of bounding boxes for each phrase. For each sentence in the test set, we predict 100 bounding boxes and use the soft token alignment prediction to rank the boxes according to the score given to the token positions that correspond to the phrase. During pre-training, given the caption “The woman wearing a blue dress standing next to the rose bush.”, MDETR would be trained to predict boxes for all referred objects such as the woman, the blue dress and the rose bush. However, for referring expressions, the task would be to only return one bounding box, which signifies the woman being referred to by the entire expression.
- Visual Question Answering

## Metrics
<img width="873" alt="image" src="https://user-images.githubusercontent.com/48170101/184163414-2f6969f1-4ea7-4fda-a75c-0caf4419517c.png">

Phrase Grounding           |  Referring
:-------------------------:|:-------------------------:
<img width="435" alt="image" src="https://user-images.githubusercontent.com/48170101/184163741-e047b2a3-7b28-49d1-afa1-93cb899193be.png"> |  <img width="424" alt="image" src="https://user-images.githubusercontent.com/48170101/184163881-03143ff3-289c-4d19-ad5a-2ce812920b7b.png">

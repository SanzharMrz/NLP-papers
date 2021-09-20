# Language Models are Open Knowledge Graphs	

The authors hypothesize that transformer based models like BERT or GPT2 have the capacity to learn and store knowledge about the domain, which can be converted to a structured knowledge graph. They go on to test this on TAC Knowledge Base Population (KBP) dataset and Wikidata.

The proposed approach has two steps Match & Map (MaMa):
![image](https://user-images.githubusercontent.com/48170101/133995257-bce628e4-b9e1-40c2-9932-de4dcf235db7.png)

_Offtop: Second step raises questions about how exactly to generate triplets_ 

Looks like they apply model prediction on some anchor entities. And then filter relationships via specified [relationships linker](https://github.com/informagi/REL). Here is a matching process by running a beam search to find triples with highest aggregate score.

![image](https://user-images.githubusercontent.com/48170101/133997701-c6e8834e-b331-4623-af40-7f622a31daf1.png)

Let's pay attention to more detailed pipeline 
![image](https://user-images.githubusercontent.com/48170101/134009528-6938a4b4-a159-4363-894f-c92d60d51224.png)

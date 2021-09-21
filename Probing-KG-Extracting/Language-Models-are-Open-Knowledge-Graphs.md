# Language Models are Open Knowledge Graphs	

Authors hypothesize that transformer based models like BERT or GPT2 have the capacity to learn and store knowledge about the domain, which can be converted to a structured knowledge graph without any fine-tuning in only forward pass.

![image](https://user-images.githubusercontent.com/48170101/134204968-6b32fd97-3a4b-4fcc-b2ee-5812c6f3af75.png)


<!-- Let's pay attention to more detailed pipeline 
![image](https://user-images.githubusercontent.com/48170101/134009528-6938a4b4-a159-4363-894f-c92d60d51224.png) -->

1. First of all, they trying to extract Nouns from sentence, for example: Dylan and songwriter here.

    ![image](https://user-images.githubusercontent.com/48170101/134206684-9d829e77-7493-400b-ac51-401359acfec6.png)
    
    __#TODO: test for more complicated sentence.__

2. Between head and tail might be a relation. So they take an attention matrix and apply some dynamic algorithm (beam search): start from initial head candidate, get argmax of this column, then go to the column with this index and repeat.  
    
    ![image](https://user-images.githubusercontent.com/48170101/134212635-7a291396-f970-4f60-b1f4-cf3dc5dff8dc.png)
    
    More detailed description:
    
    ![image](https://user-images.githubusercontent.com/48170101/134228107-63e4c3ad-1e7a-4ab0-b717-12d0c4dff28b.png)

    
    After collecting relations between entities, they apply some constraints:
    
    1. Matching degree (h, r, t) should be above > some statistic threshold
    2. Relation degree r should be above > some statistic threshold, to avoid over specified relations
    3. Skip relations which have  no meaningful interpretation [__HOW???__]
    
3. Mapping via REL and other frameworks. They demonstrate many mapping cases, like: [__In our case we can skip mapping into some KG, and use linking maye only filtering triplets?__]

    1. MAPPED FACTS IN KG SCHEMA, fully mapped h, r, t's
    2. Partially unmapped facts: at least one of h, r, and t are mapped to the KG  
    3. Completely unmapped facts: indicate all h, r, and t are not mapped to the KG schema
  
4. Insights:

    1. Larger/deeper LMs produce KGs of higher quality
    2. BERT LMs outperform GPT-2 LMs under similar model sizes
    3. The quality of unmapped facts is verified. We find 35.3% of the unmapped facts are true on Wikidata. We find 83.2% of those true facts are partially unmapped
    4. Instead of the middle layers, they use the latest. Maybe this is the point of growth, shall we try to use the middle layers with more semantic information :fire:

5. Metrics:

Probably, recall is so low, because of constraints and pos-tagging model.
![image](https://user-images.githubusercontent.com/48170101/134232047-9b303aac-4d70-47cb-a248-aa827de25be9.png)


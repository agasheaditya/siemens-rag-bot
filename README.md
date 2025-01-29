# Siemens  Assesment
You are tasked with creating an Intelligent Assistant for a technical domain (e.g., legal, manufacturing etc.). The system should: 1. Retrieve relevant documents or snippets based on user queries. 2. Generate concise, accurate, and domain-specific responses using a locally hosted small language model. 

--- 
### Working:
_The code is to build a working simple RAG based chat system which can be later utilised to create an end to end application. Here we are using manufacturing data from "data" folder in PDF format to read some documentations. 
We have used that data and reading it page wise and then saving the documents into semantics chunks in the vector DB [chroma DB]. 
Code will use sentence transformer to find out similar meaning sentences and then split/chunk it based on the meaning break or change in semantics of the sentence. 
Then we convert the text into sentence transformer embeddings and then stored it into a vector DB._

_Then we are using Llama 2 model to do the retrieval and for intelligent response generation with using top N relevent documents from Vector DB._

_Then we have asked model specifically to give answer in a QnA way so it has provided the output/response in the similar fashion with correct info as in the screenshot below._

![image](https://github.com/user-attachments/assets/74d03dd1-2a33-40c5-b778-f59d6cc4d4b9)

Also, I tried implementing some metrics system which can evaluate the recall and precision of the input, ground truth and the response from the model based on its context as we used some market proven metrics like ROUGE, BLEU, BERT Score to evaluate the responses.

* Precision and Recall at rank k measure how well the system retrieves relevant documents among the top k results.
* BLEU (Bilingual Evaluation Understudy) - Measures n-gram overlap between the generated response and the reference (ground truth).
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation) - Measures recall-based n-gram overlap (focuses on how much of the reference text is covered by the generated text).
* BERTScore - Uses semantic similarity instead of just word overlap.
--- 

### Setup and Usage: 
* Hardware used:
  - CPU: Intel i7-10750H (2.60 GHz)
  - RAM: 16 GB
  - GPU: NVIDIA GeForce RTX 1080 (6 GB)
    
* Create virtual environment
```code
virtualenv env
```

* Activate virtual environment
```code
./env/Scripts/activate
```

* Installing dependancies
```code
pip install -r requirements.txt
```
---

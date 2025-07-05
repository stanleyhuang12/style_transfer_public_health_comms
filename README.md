## Code Repo for {paper}

The goal of this project is to see if we can better understand the semantic or syntactic differences 
between regions (operationalized as 2 states) and communicative frames (see Entman 1994 & Boydstun 2014) 
among public advocacy messages. If we can better understand how messaging is tailored to certain groups, 
we can improve how generative models produce more aligned communications that is more resonant. 

**Methods**: 

We propose to do this by looking at how certain messages are represented under the hood in large 
language models (LLM). Specifically, we build probe models that examine all the hidden representations or 
activation outputs in each layer to see if there are useful structures in predicting regional differences or 
specific frames. With the probe models, we can build a steering vector that adds a bias and 
intervenes on the model. Theoretically, this should move language more towards the direction that you are predicting.
Although the literature still has to better evaluate this. 


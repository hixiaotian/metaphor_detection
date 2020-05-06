# HanCon
To be filled with more meaningfull contet! :)
https://aclweb.org/anthology/papers/W/W19/W19-0415/

# Usage 

Create an Embedding model:
EmbModel = EmbeddingModel()

Feed the model to a Predictor:
predictor = Predictor(EmbModel.emb)

Predict words:
concreteness_value = predictor.predict("word")

# Problems

- HanCon only supports fastText vectors. 
- WordEmbeddings needs to be stored localy (4.5GB). 
- WordEmbeddings are currently stored in the Project.

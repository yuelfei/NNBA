# NNBA
Recognizing Nested Named Entity based on the Neural Network Boundary Assembling Model


**Abstract**
The task to recognize named entities is often modelled as a sequence labelling process, which selects a label path whose probability is maximum for an input sentence. Because it makes the assumption that the input sentence has a flattened structure, it often fails to recognize nested named entities. In our previous work, a Boundary Assembling (BA) model was proposed. It is a cascading framework, which identifies named entity boundaries first, then assembles them into entity candidates for further assessment. This model is effective to recognize nested named entities, but still suffers from poor performance caused by the sparse feature problem. In this paper, the BA model is remodelled with the advancement of neural network, which enables the model to capture semantic information of a sentence by using word embeddings pre-trained in external resources. In our experiments, it shows an impressive improvement on the final performance, outperforming the state of the art more than 17\% in F-score.

**Keywords**
Boundary Assembling; Nested Named Entity; Information Extraction




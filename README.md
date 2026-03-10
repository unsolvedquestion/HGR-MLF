# HGR-MLF
Heterogeneous Graph Reasoning with Multi-Level Filtering for Document-Level Relation Extraction
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)

>The core challenge of Document-level Relation Extraction (DocRE) lies in handling long-range dependencies across sentences. Although heterogeneous graph reasoning has demonstrated great potential in modeling complex entity interactions, this complexity is a double-edged sword. In long documents, heterogeneous graphs introduce a large number of background vocabularies irrelevant to relation identification, resulting in a high signal-to-noise ratio (SNR) problem in the graph. Specifically, traditional graph construction methods often generate excessive false edges, leading to "over-smoothing" or "error accumulation" of semantic information during multi-layer graph convolution reasoning. Ultimately, this makes it difficult for the model to distinguish between genuine logical relations and incidental co-occurrence noise.
>
>To address this challenge, we propose a Heterogeneous Graph Reasoning with Multi-Level Filtering for Document-level Relation Extraction (HGR-MLF-DocRE), whose core idea is to implement "full lifecycle" noise blocking through a multi-level filtering mechanism. This mechanism is not a simple post-processing step but is deeply integrated with heterogeneous graph reasoning:
>
>- **Input stage**: Attention mechanisms are used to remove text spans that contribute nothing to relation judgment, ensuring the "purity" of initial node representations;
>- **Reasoning stage**: Redundant connections in heterogeneous graph reasoning are dynamically pruned through topological structure optimization (e.g., meta-path pruning), preventing inference paths from deviating from targets amid complex meta-path interactions;
>- **Output stage**: Consistency correction is applied using classification probability distributions to resolve common one-to-many or many-to-many relation conflicts in DocRE, ensuring the logical coherence of extracted relations.
>
>Experimental results show that this "filtering-guided reasoning" architecture significantly improves the accuracy of information representation, which proves the effectiveness of combining heterogeneous graph reasoning (for enriching information representation) and multi-level filtering mechanisms (for information denoising) in the DocRE task.

# Environments<br>

* Ubuntu-18.10.1(4.18.0-25-generic)<br>
* Python(3.6.8)<br>
* Cuda(10.1.243)<br>

# Dependencies<br>

* matplotlib (3.3.2)<br>
* networkx (2.4)<br>
* nltk (3.4.5)<br>
* numpy (1.19.2)<br>
* torch (1.3.0)<br>

# Data<br>

First you should get pretrained Bert_base model from [huggingface](https://github.com/huggingface/transformers) and put it into `./bert/bert-base-uncased/`. <br>
Before running our code you need to obtain the DocRED dataset from the author of the dataset, [Here](https://github.com/thunlp/DocRED).<br>
After downing DocRED, you can use `gen_data_extend_graph.py` to preprocess data for Glove-HDR-DREM and use `gen_bert_data_extend_graph.py` to preprocess data for BERT-HDR-DREM. Finally, processed data will be saved into `./prepro_data` and `./prepro_data_bert` respectively.<br> 
For the CDR, you can obtain it from https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/.
For the GDA, you can obtain it from https://bitbucket.org/alexwuhkucs/gda-extraction/src/master/.

# Run code<br>

`train.py` used to start training<br>
`test.py` used to evaluation model's performance on Dev or Test set.<br>
`Config.py` is for training Glove-based model And `Config_bert.py` is used for training Bert_based model

# Evaluation<br>

For Dev set, you can use `test.py` to evaluate you trained model.
For Test set, you should first use `test.py` to get test results which saved in `./result`, and submit it into [Condalab competition](https://competitions.codalab.org/competitions/20717).

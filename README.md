# COLING20: Incremental Neural Lexical Coherence Modeling
### [Sungho Jeon](https://sdeva14.github.io/) and [Michael Strube](https://www.h-its.org/people/prof-dr-michael-strube/)
#### [NLP Lab, Heidelberg Institute for Theoretical Studies (HITS)](https://www.h-its.org/research/nlp/people/)

This project contains a python implementation for the COLING20 paper whose title is "Incremental Neural Lexical Coherence Modeling".
## Requirements

#### Conda environment
We recommend using Conda environment for setup. It is easy to build an environment by the provided environment file. It is also possible to setup manually by considering the information in "spec-file.txt". 

Our environment file is built based on CUDA9 driver and corresponding libraries, thus it should be managed by the target GPU environment. Otherwise, GPU flag should be disabled as a library. For the variation of XLNet, we use Transformers library implemented by Huggingface (Wolf et al, 2019).

    conda create --name py3_torch_cuda9 --file spec-file.txt
    source activate py3_torch_cuda9
    pip install pytorch-transformers==2.3.0

#### Dataset and materials
Dataset and pretrained embedding cannot be attached in the submission due to large size, thus it should be downloaded from the Github of previous work.

- Dataset: The GCDC dataset is available as guided their GitHub with the license for Yahoo Answers corpus. This license can be requested free of charge for research purposes. The TOEFL dataset is available according to the link in the original paper (Blanchard et al. 2013) with LDC license. The location of the target dataset should be configured in "build_config.py" with "--data_dir" option. A CV partition can be created by our implementation using --is_gen_cv option in build_config.py. The indexes of CV partitions are attached for two datasets, "toefl_ids_cv5.tar.gz" and "gcdc_ids_cv10.tar.gz", respectively.

GCDC dataset link: https://github.com/aylai/GCDC-corpus

TOEFL dataset link: https://catalog.ldc.upenn.edu/LDC2014T06

For our model and the experiments on both datasets, we use the 100-dimensional pre-trained embedding model on Google News, Glove (Pennington, Socher, and Manning 2014). We use pretrained model "XLNet-base" for the variation of XLNet.

Glove link: https://nlp.stanford.edu/projects/glove/

XLNet link: https://github.com/huggingface/transformers/

## Run Models
#### Basic run
A basic run is performed by "main.py" with configuration options by providing in terminal or modifying "build_config.py" file.
Detail information about the configuration can be found in the "build_config.py" ("ilcr_lcsd" is our model which uses a semantic distance vector).

	Examples for execution (assume that a data path is given in build_config.py).
    For TOEFL) python main.py --essay_prompt_id_train 3 --essay_prompt_id_test 3 --target_model ilcr_lcsd

    For GCDC) python main.py --gcdc_domain Clinton --target_model ilcr_lcsd

#### The list of models
	conll17: The automated essay scoring model in Dong et al. (2017)
	emnlp18: The coherence model in Mesgar and Strube (2018)
	sent_avg: The baseline which encodes sentences individually, then averages all sentence representations
	doc_avg: The baseline which encodes a document at once, then averages all representations
	inc_lexi: Our model which represents lexical coherence

#### Pre-defined configuration
For the convenient reproduction, we provide four configuration examples, a configuration with RNN models (e.g., "toefl_build_config.py") and a configuration with XLNet model (e.g., "toefl_xlnet_build_config.py") for the two datasets.
The location of the dataset and pretrained embedding layer should be managed properly in "build_config.py".

Note that additional parameters for baseline models should be configured as target models as described in the literatures

## Model Parameters
We describe model parameters as follows, and more details can be found in each configuration files, e.g.,)"build_config.py", "toefl_xlnet_build_config.py".

| Dataset  | Learning rate | Droptout | Emb dim | RNN cell size | Batch size | Eps | Conv size | Conv pad |  Sent-level pool size |
| ------------- | :---: | :---: | :---: |    :---: |  :---: |  :---: |  :---: | :---: | :---: |  
| GCDC  | 0.001  | 0.5 | 100 | 150 | 32 | 1e-6 | (5, 1) | 2 | 5 |
| TOEFL  | 0.003  | 0.1 | 100 | 150 | 32 | 1e-6 | (3, 2) | 1 | 5 |

Note that we apply two different parameters for the convolutional layer as prompts in datasets: we apply (kernel:5,stride:1,pad:2) for domains Clinton and Enron on GCDC, and we apply (kernel:3,stride:2,pad:1) for prompt 1 and prompt 5 on TOEFL.

## State of the art in TOEFL
We also compare with the state of the art on TOEFL, Nadeem et al.(2019). We notice that the reported performance in Nadeem et al.(2019) cannot be compared with previous work due to a different experimental setup; they filter out the more than 7.5% of sentences whose length is longer than a length threshold, and they evaluate performance without a cross validation. To ensure a fair comparison, we only modified the experimental setup in their implementation.

We attached both of their original implementation and our modified version: "aes_bea19_origin.tar.gz" and "BERT_BCA_train_modified.py" in "aes_bea19_compare.tar.gz", respectively. We only modified their implementation to evaluate performance in a 5-fold cross validation setting, thus it still filters sentences by their length threshold. We observe that the performacne is lower when we do not filter sentences as previous work does. See their Github page for more details.

link: https://github.com/Farahn/AES

## Acknowledge
This implementation was possible thanks to many shared implementations. We describe an original source link at the first line of codes if we use theirs.

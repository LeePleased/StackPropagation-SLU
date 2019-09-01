# A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding

This repository contains the PyTorch implementation of the paper: **A Stack-Propagation Framework with 
Token-Level Intent Detection for Spoken Language Understanding**. If you use any source codes or the datasets 
included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

<pre>
@inproceedings{qin2019stack,
  title={A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding},
  author={Libo Qin, Wanxiang Che, Yangming Li, Haoyang Wen, Ting Liu},
  booktitle={Proceedings of the Empirical Method for Natural Language Understanding (EMNLP)},
  year={2019}
}
</pre>

In the following, we will guide you how to use this repository step by step.

## Preparation

Our code is based on PyTorch 1.1 and runnable for both windows and ubuntu server. Required python packages:
    
> + numpy==1.16.2
> + tqdm==4.32.2
> + scipy==1.2.1
> + torch==1.1.0
> + ordered-set==3.1.1

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## Running Code

The script **train.py** acts as a main function to the project. For reproducing the results reported in our
paper.

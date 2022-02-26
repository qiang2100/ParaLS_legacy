# ParaLS: ParaLS: Paraphraser-based Lexical Substitution

Lexical substitution (LS) aims at finding appropriate substitutes for a target word in a sentence. Recently, BERT-based LS methods have made remarkable progress, which generates substitute candidates for a target word directly based on its context. However, it overlooks the substitution's impact on the overall meaning of the sentence. In this paper, we try to generate the substitute candidates from a paraphraser. Considering the generated paraphrases from a paraphraser contain variations in word choice and preserve the sentence's meaning, we propose a simple decoding method that focuses on the variations of the target word during decoding, and leverage it to propose a new LS approach ParaLS. Experimental results show that ParaLS improves the F1 score from 18.4 to 28.7 on the up-to-date benchmark compared with the state-of-the-art BERT-based LS method.



# Requirements and Installation

* [PyTorch](http://pytorch.org/) version = 1.8.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)


# Citation

Please cite as:

``` bibtex
@inproceedings{QiangParaLS,
  title = {ParaLS: Paraphraser-based Lexical Substitution
Abstract},
  author = {Jipeng Qiang, Yun Li, Yunhao Yuan, Yi Zhu, Xindong },
  booktitle = {****},
  year = {2022},
}
```

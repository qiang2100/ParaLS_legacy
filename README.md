# ParaLS: ParaLS: Paraphraser-based Lexical Substitution

Lexical substitution (LS) aims at finding appropriate substitutes for a target word in a sentence. Recently, BERT-based LS methods have made remarkable progress, which generates substitute candidates for a target word directly based on its context. However, it overlooks the substitution's impact on the overall meaning of the sentence. In this paper, we try to generate the substitute candidates from a paraphraser. Considering the generated paraphrases from a paraphraser contain variations in word choice and preserve the sentence's meaning, we propose a simple decoding method that focuses on the variations of the target word during decoding, and leverage it to propose a new LS approach ParaLS. Experimental results show that ParaLS improves the F1 score from 18.4 to 28.7 on the up-to-date benchmark compared with the state-of-the-art BERT-based LS method.



# Requirements and Installation

*  Our code is based on [Fairseq](https://github.com/pytorch/fairseq) version=10.2
* [PyTorch](http://pytorch.org/) version = 1.8.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

## Step 1: Downlaod the pretrained paraphraser modeling

You need to download the paraphraser from [here](https://drive.google.com/file/d/1o5fUGJnTxMe9ASQWTxIlbWmbEqN_RQ6D/view?usp=sharing), and put it into folder "checkpoints/⁨para⁩/transformer/⁩"

## Step 2: Run our code 

(1) run ParaLS for lexical substitute dataset SWORDS

input "sh run_Swords.sh"

(2) run ParaLS for lexical simplification dataset 

input "sh run_LSPara_NNSeval.sh"

(3) run ParaLS for one example

input "python Paraphraser.py"


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

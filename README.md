# Few Shot Knowledge Transfer

This is the KD+AT few shot baselines (Knowledge Distillation + Attention Transfer) for the NeurIPS 2019 spotlight paper "Zero-shot Knowledge Transfer via Adversarial Belief Matching" [see arxiv](https://arxiv.org/abs/1905.09768)

With this code you should be able to reproduce KD+AT accuracies in Figure 2 and Table 1 of the paper.

For the main code of the paper see [this repo](https://github.com/polo5/ZeroShotKnowledgeTransfer).


## Environment
- Python 3.6
- pytorch 1.0.0 (both cpu and gpu version tested)
- tensorboard 1.7.0 (for logging, needs tensorflow, other versions probably ok)

## Run few shot knowledge transfer
1. Pretrain a teacher for the dataset/architecture you want (or download some of mine [here](https://drive.google.com/drive/folders/1lLgAndtJGUOUWvFGC8f1BFA5RIgyEfct?usp=sharing))
2. Make sure you have the same folder structure as in the link above, i.e. Pretrained/{dataset}/{architecture}/last.pth.tar
3. Edit the paths in e.g. scripts/CIFAR10/WRN-40-2_WRN-16-1/main0.sh and run it

## Cite
If you use this work please consider citing:
```
@article{Micaelli2019ZeroShotKT,
  author    = {Paul Micaelli and
               Amos J. Storkey},
  title     = {Zero-shot Knowledge Transfer via Adversarial Belief Matching},
  journal   = {CoRR},
  volume    = {abs/1905.09768},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.09768},
  archivePrefix = {arXiv},
  eprint    = {1905.09768},
  timestamp = {Wed, 29 May 2019 11:27:50 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-09768},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
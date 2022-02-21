# Quadric hypersurface intersection for manifold learning in feature space

This repository contains a PyTroch implementation of the algorithm presented in the paper Quadric hypersurface intersection for manifold learning in feature space
[https://arxiv.org/abs/2102.06186](https://arxiv.org/abs/2102.06186)

## Prerequisites

You will need python 3.6+ and the following packages

- `numpy`
- `pytorch 1.7`
- `sklearn 0.24.2` if you need kernel models
- `matplotlib` for graphs plotting in notebooks with examples
- `tqdm`
- `scipy` for Seam_line_exaple notebook
- `pandas` for tables building


## Quadric models

`quadrics` folder contains implementation of quadric intersection manifold

- `model.py`: implementation of quadrics as `torch.nn.Module`.
- `quadrics_wrapper`: a wrapper for the model with `sklearn-like` interface including trainer implementation

## Examples

The examples folder contain Jupyter notebooks with the following examples:

- `Seam_line_example`: a toy example of approximation of the tennis ball seam line using quadrics
- `IR_full`: identification rate calculation for cplfw dataset with megaface distractors, with different methods features calculation  
- `ood_test`: out-of-distribution test, find distance between of outliers and given datasets features distribution in roc-auc score metric

## Data
Out-of-distribution 
For downloading dataset embeddings follow the next link and put file to the project folder:
https://drive.google.com/file/d/1o7uUkkbIvHKEMSAcQV9rs9L0HUkY7dsh/view?usp=sharing
To extract archive and to prepare folders for future calculation execute next file:

- `./prepare_folders.sh`

Initial images dataset you may find there: [GitHub Pages](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_). To get final embeddings resnet-50 nettwork with ArcFace loss was used. For ood tests we provide embeddings datatset of  Also, in `image_embeddings/labels/cplfw_outliers_labels.json` you may find cplfw outliers images detected by eyes. 

## Citation
```
@article{pavutnitskiy2021quadric,
  title={Quadric hypersurface intersection for manifold learning in feature space},
  author={Pavutnitskiy, Fedor and Ivanov, Sergei O and Abramov, Evgeny and Borovitskiy, Viacheslav and Klochkov, Artem and Vialov, Viktor and Zaikovskii, Anatolii and Petiushko, Aleksandr},
  journal={arXiv preprint arXiv:2102.06186},
  year={2021}
}
```

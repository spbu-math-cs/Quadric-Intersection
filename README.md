# Quadric hypersurface intersection for manifold learning in feature space

This repository contains a PyTroch implementation of the algorithm presented in the paper Quadric hypersurface intersection for manifold learning in feature space
[https://arxiv.org/abs/2102.06186](https://arxiv.org/abs/2102.06186)

## Prerequisites
TODO: add requierements

## Quadric models

`quadrics` folder contains implementation of quadric intersection manifold

- `model.py`: implementation of quadrics as `torch.nn.Module`.
- `quadrics_wrapper`: a wrapper for the model with `sklearn-like` interface including trainer implementation

## Examples

The examples folder contain Jupyter notebooks with the following examples:

- `Seam_line_example`: a toy example of approximation of the tennis ball seam line using quadrics
- `ir_toy_example`: TODO: write description

## Data
TODO: write about datasets

## Citation
```
@article{pavutnitskiy2021quadric,
  title={Quadric hypersurface intersection for manifold learning in feature space},
  author={Pavutnitskiy, Fedor and Ivanov, Sergei O and Abramov, Evgeny and Borovitskiy, Viacheslav and Klochkov, Artem and Vialov, Viktor and Zaikovskii, Anatolii and Petiushko, Aleksandr},
  journal={arXiv preprint arXiv:2102.06186},
  year={2021}
}
```

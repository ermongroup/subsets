# Reparameterizable Subset Sampling via Continuous Relaxations

This repo contains the code for the paper [Reparameterizable Subset Sampling via Continuous Relaxations](https://arxiv.org/abs/1901.10517),
which allows you to include subset sampling as a layer in a neural network.
This is useful whenever you want to select a discrete number of elements, such as in
dynamic feature selection or k-nearest neighbors.
This repo contains the experiments for learning feature selectors for explainability,
training a deep stochastic k-NN model, and training a parametric t-SNE model using subset sampling.

Supports the following libraries:
- PyTorch (`SubsetOperator` in `subsets/knn/sorting_operator.py`) versions of the differentiable subset sampler are available.
- TensorFlow (`sample_subset` in `subsets/sample_subsets.py`)

To setup, please create a new Python virtualenv with Python 3.6, activate it,
navigate to this directory (containing `setup.py`) and run
`pip install -e .`

To run the experiments, navigate to the `experiments/` folder and run the
corresponding scripts.

If you find this code useful, please cite
```
@article{xie2019subsets,
  author    = {Sang Michael Xie and Stefano Ermon},
  title     = {Reparameterizable Subset Sampling via Continuous Relaxations},
  journal   = {International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019}
}
```

# 479-Project

This project attempts to reproduce results of [Pure Noise to the Rescue of Insufficient Data](https://arxiv.org/abs/2112.08810) by Shiran Zada, Itay Benou, and Michal Irani. [Previous work](https://zenodo.org/records/8173763) by Seungjae Ryan Lee and Seungmin Brian Lee has also attempted to do so.

## Requirements

Required libraries are listed in `requirements.txt`. On Python versions prior to 3.11, `toml` is also required as an alternative to the standard library `tomllib`.

## Usage

This program is primarily intended to be ran through the command line.

```bash
python main.py --model=johnson,waltz --dataset=cifar10lt
```

The following arguments are required:
- `--model=<VALUE>`: The model configuration to use. Found at `configs/model-<VALUE>.toml`. More than one model may be specified as a comma-separated list.
- `--dataset=<VALUE>`: The dataset configuration to use. Found at `configs/dataset-<VALUE>.toml`

The following arguments are optional:
- `--runs=<VALUE>`: Indicates the number of runs to perform. Helps with datasets where many random decisions are made. Defaults to `1`.
- `--dry-run`: Indicates that a dry run should be performed. The dataset and model are compiled, but training and evaluation are skipped.

### Notes on Usage

The construction of Long-Tail dataset variants is **non-deterministic**. In order to properly test two models off the same dataset, you **must** specify all of those models in the same program run.

## Default Configurations

We provide a number of configurations for preliminary tests. Users are encouraged to create their own.

### Datasets

We provide configurations for the following datasets:

| Dataset | Description |
| ------- | ----------- |
| `cifar10` | The CIFAR-10 dataset we know and love. |
| `cifar10lt` | The CIFAR-10 dataset with samples discarded at random. The largest class will have all of its samples, and the smallest class will have 10% of its samples. The class sizes will be calculated as a linear function. |
| `cifar100` | The CIFAR-100 dataset we know and love. |
| `cifar100lt` | The same technique is applied to the CIFAR-100 dataset as with `cifar10lt` |

### Models

We provide configurations for the following models:

| Model | Description |
| ----- | ----------- |
| `johnson` | A simple model consisting of a Conv2D (with ReLU) and BatchNorm layer. Outputs determined via softmax. Reasonable hyperparameters for quick testing. |
| `waltz` | Equivalent to `johnson`, but applies the OPeN dataset augmenting and uses DAR-BN instead of Batch Normalization. 50% of oversampling is replaced with noise. |
| `reynolds` | Like `johnson`, but with early stopping. |
| `chalamet` | Like `waltz`, but with early stopping. |
| `stewart` | Equivalent to `johnson`, but performs oversampling on minor classes. |
| `de-niro` | Equivalent to `waltz`, but only adds noise and does not oversample. |

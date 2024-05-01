# 479-Project

This project attempts to reproduce results of [Pure Noise to the Rescue of Insufficient Data](https://arxiv.org/abs/2112.08810) by Shiran Zada, Itay Benou, and Michal Irani. [Previous work](https://zenodo.org/records/8173763) by Seungjae Ryan Lee and Seungmin Brian Lee has also attempted to do so.

## Usage

This program is primarily intended to be ran through the command line.

```bash
python main.py --model=timothee --dataset=cifar10lt
```

The following arguments are required:
- `--model=<VALUE>`: The model configuration to use. Found at `configs/model-<VALUE>.toml`.
- `--dataset=<VALUE>`: The dataset configuration to use. Found at `configs/dataset-<VALUE>.toml`

The following arguments are optional:
- (None currently.)

More arguments may be added as functionality expands.


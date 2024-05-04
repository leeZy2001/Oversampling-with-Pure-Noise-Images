"""
A testing utility for OPeN and DAR-BN designed for a command-line
interface.
"""

import sys
import os

# Only allow Tensorflow to print errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from parse import parse_args, parse_toml
from model.builder import build_model, build_optimizer, build_callbacks
from dataset import build_dataset, augment_dataset


def main():
    """
    Configures and tests a model on a dataset according to command-line
    arguments.
    """
    # Looking for mandatory arguments.
    args = parse_args(sys.argv[1:])
    dataset_config, model_config = get_configs_from(args)

    # Constructing the dataset.
    dataset_block = build_dataset(dataset_config["dataset"])
    training = dataset_block["training"]
    testing = dataset_block["testing"]
    num_classes = dataset_block["num_classes"]

    # Getting the shape
    for sample in training:
        shape = sample["image"].shape
        break

    model = build_model(shape, num_classes, model_config["model"])
    optimizer = build_optimizer(model_config["model"])
    if "callbacks" in model_config["model"]:
        callbacks = build_callbacks(model_config["model"])
    else:
        callbacks = []

    model.compile(
        optimizer=optimizer, loss=model_config["model"]["loss"], metrics=["accuracy"]
    )

    model.fit(
        training,
        validation_split=model_config["hyperparams"]["validation_split"],
        epochs=model_config["hyperparams"]["epochs"],
        batch_size=model_config["hyperparams"]["batch_size"],
        callbacks=callbacks,
    )

    results = model.evaluate(
        testing,
        batch_size=model_config["hyperparams"]["batch_size"],
    )

    print(results)


def get_configs_from(args):
    errors = []
    if "dataset" not in args:
        errors.append("Missing required parameter `--dataset=<dataset>`")
    if "model" not in args:
        errors.append("Missing required parameter `--model=<model>`")

    if len(errors) > 0:
        raise KeyError(*errors)

    # Ensuring that configurations exist.
    dataset_path = "configs/dataset-" + args["dataset"] + ".toml"
    model_path = "configs/model-" + args["model"] + ".toml"
    if not os.path.exists(dataset_path):
        errors.append(f"Unable to locate dataset configuration at [{dataset_path}].")
    if not os.path.exists(model_path):
        errors.append(f"Unable to locate model configuration at [{model_path}].")

    if len(errors) > 0:
        raise KeyError(*errors)

    # Loading configurations.
    try:
        dataset_config = parse_toml(dataset_path)
    except Exception as e:
        errors.append(e)
    try:
        model_config = parse_toml(model_path)
    except Exception as e:
        errors.append(e)

    if len(errors) > 0:
        raise ValueError(*errors)

    return dataset_config, model_config


if __name__ == "__main__":
    main()

"""
A testing utility for OPeN and DAR-BN designed for a command-line
interface.
"""

import sys
import os
import tensorflow as tf

# Only allow Tensorflow to print errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from parse import parse_args, parse_toml
from dataset import build_dataset


def main():
    """
    Configures and tests a model on a dataset according to command-line
    arguments.
    """
    cannot_continue = False

    # Looking for mandatory arguments.
    args = parse_args(sys.argv[1:])
    if "dataset" not in args:
        print("Missing required parameter `--dataset=<dataset>`")
        cannot_continue = True
    if "model" not in args:
        print("Missing required parameter `--model=<model>`")
        cannot_continue = True

    if cannot_continue:
        print("An irrecoverable error occurred and the program must abort.")
        return

    # Ensuring that configurations exist.
    dataset_path = "configs/dataset-" + args["dataset"] + ".toml"
    model_path = "configs/model-" + args["model"] + ".toml"
    if not os.path.exists(dataset_path):
        print(f"Unable to locate dataset configuration at [{dataset_path}].")
        cannot_continue = True
    if not os.path.exists(model_path):
        print(f"Unable to locate model configuration at [{model_path}].")
        cannot_continue = True

    if cannot_continue:
        print("An irrecoverable error occurred and the program must abort.")
        return

    # Loading configurations.
    try:
        dataset_config = parse_toml(dataset_path)
    except:
        print(f"The dataset configuration at [{dataset_path}] was malformed.")
        cannot_continue = True
    try:
        model_config = parse_toml(model_path)
    except:
        print(f"The model configuration at [{model_path}] was malformed.")
        cannot_continue = True

    if cannot_continue:
        print("An irrecoverable error occurred and the program must abort.")
        return

    # Constructing the dataset.
    dataset_block = build_dataset(dataset_config["dataset"])
    training, validation = tf.keras.utils.split_dataset(dataset_block["training"], model_config["hyperparams"]["training_split"])
    testing = dataset_block["testing"]
    num_classes = dataset_block["num_classes"]


    # Batching the dataset.
    batch_size = model_config["hyperparams"]["batch_size"]
    training.batch(batch_size)
    validation.batch(batch_size)
    testing.batch(batch_size)

    # Getting the shape 
    for batch in training:
        shape = batch['image'].shape
        break
    # model = build_model(shape, num_classes, model_config["model"])
    # TODO: Train the model.
    # TODO: Report statistics???


if __name__ == "__main__":
    main()

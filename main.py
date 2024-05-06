"""
A testing utility for OPeN and DAR-BN designed for a command-line
interface.
"""

import sys
import os
import time

from parse import parse_args, parse_toml
from model.builder import build_model, build_optimizer, build_callbacks
from dataset import build_dataset, augment_dataset, into_workable
import tensorflow as tf


def full_evaluation_cycle(args, dataset_config, model_configs):
    print(f"Building dataset [{dataset_config['dataset']['name']}]...", end="", flush=True)
    start = time.perf_counter()

    full_training, testing, num_classes, shape = get_dataset_from_config(dataset_config)

    elapsed = time.perf_counter() - start
    print("%.3fs" % elapsed)

    results = {}
    for model_config in model_configs:
        model_name = model_config["model"]["name"]
        print(f"Building model [{model_name}]...", end="", flush=True)
        start = time.perf_counter()

        model = build_model(shape, num_classes, model_config["model"])
        optimizer = build_optimizer(model_config["model"])
        if "callbacks" in model_config["model"]:
            callbacks = build_callbacks(model_config["model"])
        else:
            callbacks = []

        model.compile(
            optimizer=optimizer, loss=model_config["model"]["loss"], metrics=["accuracy"]
        )
        elapsed = time.perf_counter() - start
        print("%.3fs" % elapsed)

        if "dataset" in model_config and "augmenting" in model_config["dataset"]:
            print(f"Augmenting dataset for model [{model_name}]...", end="", flush=True)
            start = time.perf_counter()

            personal_training = augment_dataset(full_training, num_classes, shape, model_config["dataset"]["augmenting"])

            elapsed = time.perf_counter() - start
            print("%.3fs" % elapsed)
        else:
            personal_training = full_training

        print(f"Splitting training and validation for model [{model_name}]...", end="", flush=True)
        start = time.perf_counter()

        training, validation = tf.keras.utils.split_dataset(personal_training, right_size=model_config["hyperparams"]["validation_split"])

        elapsed = time.perf_counter() - start
        print("%.3fs" % elapsed)

        if "dry-run" in args:
            continue

        batch_size = model_config["hyperparams"]["batch_size"]
        print(f"Training model [{model_name}]...")

        model.fit(
            into_workable(training, batch_size=batch_size),
            validation_data=into_workable(validation, batch_size=batch_size),
            epochs=model_config["hyperparams"]["epochs"],
            callbacks=callbacks,
        )

        results[model_name] = {}
        for name, dataset in testing.items():
            print(f"Evaluating model [{model_name}] on testing dataset [{name}]...")
            results[model_name][name] = model.evaluate(into_workable(dataset, batch_size=batch_size))[1]

    if "dry-run" in args:
        print("Dry run complete. Program exiting.")
        return

    print(f"{'Model':^16}", end="")
    dataset_labels = [name for name in testing.keys()]
    for label in dataset_labels:
        print(f" | {label:<12}", end="")
    print()
    for model_name, model_results in results.items():
        print(f"{model_name:<16}", end="")
        for ds in dataset_labels:
            print(f" | {model_results[ds]:>8.2%}", end="")
        print()
    

def main():
    """
    Configures and tests a model on a dataset according to command-line
    arguments.
    """
    args = parse_args(sys.argv[1:])
    verify_necessary_args_are_present(args)

    dataset_config = get_dataset_config(args)
    model_configs = get_all_model_configs(args)

    runs = args["runs"] if "runs" in args else 1
    for run in range(runs):
        print(f"Performing run {run+1} of {runs}")
        full_evaluation_cycle(args, dataset_config, model_configs)


def verify_necessary_args_are_present(args):
    errors = []
    if "dataset" not in args:
        errors.append("Missing required parameter `--dataset=<dataset>`")
    if "model" not in args:
        errors.append("Missing required parameter `--model=<model>`")
    if len(errors) > 0:
        raise KeyError(*errors)


def get_dataset_config(args):
    dataset_path = "configs/dataset-" + args["dataset"] + ".toml"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Unable to locate dataset configuration at [{dataset_path}].")
    return parse_toml(dataset_path)


def get_all_model_configs(args):
    model_names = args["model"].split(",")
    configs = []
    for model_name in model_names:
        model_path = "configs/model-" + model_name + ".toml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Unable to locate model configuration at [{model_path}].")
        configs.append(parse_toml(model_path))
    return configs


def get_dataset_from_config(dataset_config):
    dataset_block = build_dataset(dataset_config["dataset"])
    full_training = dataset_block["training"]
    testing = dataset_block["testing"]
    num_classes = dataset_block["num_classes"]
    for sample in full_training:
        shape = sample["image"].shape
        break
    return full_training, testing, num_classes, shape


if __name__ == "__main__":
    main()

import sys
import os

# Only allow Tensorflow to print errors.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from parse import parse_args, parse_toml
from model.builder import build_model, build_optimizer


def main():
    cannot_continue = False

    args = parse_args(sys.argv[1:])
    if not "dataset" in args:
        print("Missing required parameter `--dataset=<dataset>`")
        cannot_continue = True
    if not "model" in args:
        print("Missing required parameter `--model=<model>`")
        cannot_continue = True

    if cannot_continue:
        print("An irrecoverable error occurred and the program must abort.")
        return

    dataset_path = "configs/dataset-" + args["dataset"] + ".toml"
    model_path = "configs/model-" + args["model"] + ".toml"
    if not os.path.exists(dataset_path):
        print("Unable to locate dataset configuration.")
        cannot_continue = True
    if not os.path.exists(model_path):
        print("Unable to locate model configuration.")
        cannot_continue = True

    if cannot_continue:
        print("An irrecoverable error occurred and the program must abort.")
        return

    dataset_config = parse_toml(dataset_path)
    model_config = parse_toml(model_path)
    # TODO: We should also check that the TOML is not malformed.

    # TODO: Call `build_dataset` from the `dataset` module.
    # TODO: Call `build_model` from the `model.builder` module.
    # TODO: Train the model.
    # TODO: Report statistics???


if __name__ == '__main__':
    main()

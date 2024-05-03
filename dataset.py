"""
Utilities for generating and augmenting datasets.
"""

import numpy as np
import random
from typing import Any
import tensorflow_dataset

def get_scaling_function(fn_config: dict[str, Any], num_classes: int):
    match fn_config["function"]:
        case "linear":
            smallest_size = fn_config["smallest_size"]
            return lambda n: smallest_size + (1 - smallest_size) * n / num_classes
    return None


def build_dataset(dataset_config: dict[str, Any]):
    """Constructs a dataset from a configuration."""
    datasets, info = tfds.load(dataset_config["source"], with_info=True)
    training_dataset = datasets["train"]
    testing_dataset = datasets["testing"]
    if "trimming" in dataset_config:
        organized = {}
        for label in info.features['label'].names:
            organized[label] = []
        for example in training_dataset:
            image, label = example['image'], example['label']
            organized[label].append(image)
        scaling_fn = get_scaling_function(dataset_config["trimming"], info.features['label'].num_classes)
        max_size = max(len(l) for l in organized.values())
        labels = [*organized.keys()]
        if dataset_config["trimming"]["shuffle_classes"]:
            random.shuffle(labels)
        for index, label in enumerate(labels):
            if dataset_config["trimming"]["shuffle_files"]:
                random.shuffle(organized[label])
            upper = int(max_size * scaling_fn(index))
            organized[label] = organized[label][:upper]
        combined = [(label, image) for image in organized[label] for label in organized.keys()]
        random.shuffle(combined)
        images = []
        labels = []
        for label, image in combined:
            images.append(image)
            labels.append(label)
        image_tensor = tf.constant(images)
        label_tensor = tf.constant(labels)
        training_dataset = Dataset.from_tensor_slices((image_tensor, label_tensor))
    return {
        "training": training_dataset,
        "testing": testing_dataset,
        "num_classes": info.features['label'].num_classes
    }


def generate_noise_images(
    num_images: int, image_shape, mean: float, standard_deviation: float
):
    """Produces normally-distributed random noise images."""
    return np.clip(
        np.random.normal(
            loc=mean, scale=standard_deviation, size=(num_images, *image_shape)
        ),
        0,
        1,
    )

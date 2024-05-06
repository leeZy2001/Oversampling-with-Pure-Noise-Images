"""
Utilities for generating and augmenting datasets.
"""

import numpy as np
import random
from typing import Any
import tensorflow_datasets as tfds
import tensorflow as tf


def get_scaling_function(fn_config: dict[str, Any], num_classes: int):
    match fn_config["function"]:
        case "linear":
            smallest_size = fn_config["smallest_size"]
            return lambda n: smallest_size + (1 - smallest_size) * n / (num_classes - 1)
    return None


def build_dataset(dataset_config: dict[str, Any]):
    """Constructs a dataset from a configuration."""
    datasets, info = tfds.load(dataset_config["source"], with_info=True)
    training_dataset = datasets["train"]
    testing_dataset = datasets["test"]
    if "trimming" in dataset_config:
        training_dataset = trim_dataset(
            training_dataset,
            info.features["label"].num_classes,
            dataset_config["trimming"],
        )
    return {
        "training": training_dataset,
        "testing": testing_dataset,
        "num_classes": info.features["label"].num_classes,
    }


def augment_dataset(dataset, num_classes, shape, augment_config):
    """Applies oversampling and the OPeN technique according to the config."""
    pad_to_percent = augment_config["pad_to_percent"]
    replace_with_noise = augment_config["replace_with_noise"]

    organized = sort_dataset(dataset, num_classes)
    mean, stdev = compute_statistics_from_sorted(organized)
    pad_size = int(max(len(l) for l in organized.values()) * pad_to_percent)
    for label in organized.keys():
        pad_amount = pad_size - len(organized[label])
        if pad_amount <= 0:
            continue
        noise_to_add = int(pad_amount * replace_with_noise)
        oversamples_to_add = pad_amount - noise_to_add
        padding = [random.choice(organized[label]) for _ in range(oversamples_to_add)]
        padding.extend(generate_noise_images(noise_to_add, shape, mean, stdev))
        organized[label].extend(padding)
    return flatten_dataset(organized)


def compute_statistics_from_sorted(organized):
    """Given a dataset in the form provided by `sort_dataset`, computes the mean and standard deviation."""
    samples = [image for images in organized.values() for image in images]
    tensor = tf.cast(tf.convert_to_tensor(samples), tf.float32)
    return tf.math.reduce_mean(tensor), tf.math.reduce_std(tensor)


def trim_dataset(dataset, num_classes, trimming_config):
    """Trims a dataset according to the config, in order to construct long-tail variants."""
    organized = sort_dataset(dataset, num_classes)
    scaling_fn = get_scaling_function(trimming_config, num_classes)
    max_size = max(len(l) for l in organized.values())
    labels = [*organized.keys()]
    if trimming_config["shuffle_classes"]:
        random.shuffle(labels)
    for index, label in enumerate(labels):
        if trimming_config["shuffle_files"]:
            random.shuffle(organized[label])
        upper = int(max_size * scaling_fn(index))
        organized[label] = organized[label][:upper]
    return flatten_dataset(organized)


def flatten_dataset(organized, shuffle=True):
    """Inverts the process done by `sort_dataset`."""
    combined = [
        (label, image) for label, images in organized.items() for image in images
    ]
    if shuffle:
        random.shuffle(combined)
    ids = []
    images = []
    labels = []
    for idnum, (label, image) in enumerate(combined):
        ids.append(idnum)
        images.append(image)
        labels.append(label)
    id_tensor = tf.constant(ids)
    image_tensor = tf.convert_to_tensor(images)
    label_tensor = tf.constant(labels)
    return tf.data.Dataset.from_tensor_slices(
        {"id": id_tensor, "image": image_tensor, "label": label_tensor}
    )


def sort_dataset(dataset, num_classes):
    """Converts a dataset object into a dictionary mapping each label to a list of its samples."""
    organized = {}
    for label in range(num_classes):
        organized[label] = []
    for example in dataset:
        image, label = example["image"], example["label"]
        organized[int(label)].append(image)
    return organized


def generate_noise_images(
    num_images: int, image_shape, mean: float, standard_deviation: float
):
    """Produces normally-distributed random noise images."""
    return [
        tf.cast(np.clip(
            np.random.normal(loc=mean, scale=standard_deviation, size=image_shape),
            0,
            255,
        ), tf.uint8)
        for _ in range(num_images)
    ]


def into_workable(dataset, batch_size=None, cast=tf.float32):
    """Converts a dataset into a tuple-formatted dataset, as expected by `model.fit`."""
    images = []
    labels = []
    for sample in dataset:
        images.append(tf.cast(sample['image'], cast))
        labels.append(sample['label'])
    image_tensor = tf.convert_to_tensor(images)
    label_tensor = tf.convert_to_tensor(labels)
    new_dataset = tf.data.Dataset.from_tensor_slices((image_tensor, label_tensor))
    return new_dataset.batch(batch_size) if batch_size is not None else new_dataset

"""
Builder functions for constructing models and related components from config files.
"""

from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.models import Model

from model.darbn import CustomDARBN


def build_model(input_shape, num_classes, model_config):
    """
    Constructs a model from a configuration and related dataset
    information.
    """
    inputs = layers.Input(shape=input_shape)
    last_layer = inputs
    for layer_config in model_config["layers"]:
        last_layer = build_layer(**layer_config)(last_layer)
    outputs = layers.Dense(num_classes, **model_config["output"])(last_layer)
    model = Model(name=model_config["name"], inputs=inputs, outputs=outputs)
    return model


def build_optimizer(model_config):
    """Constructs an optimizer from a configuration."""
    match model_config["optimizer"]:
        case "adam":
            return optimizers.Adam()
    return None


def build_callbacks(model_config):
    model_callbacks = []
    for callback_cfg in model_config["callbacks"]:
        model_callbacks.append(build_single_callback(**callback_cfg))
    return model_callbacks


def build_single_callback(callback_type, **options):
    match callback_type:
        case "early_stopping":
            return callbacks.EarlyStopping(**options)
    return None


def build_layer(layer_type, **options):
    """Constructs a layer from the type and set of options."""
    match layer_type:
        case "conv2d":
            return layers.Conv2D(**options)
        case "relu":
            return layers.ReLU(**options)
        case "darbn":
            return CustomDARBN(**options)
        case "flatten":
            return layers.Flatten(**options)
    return None

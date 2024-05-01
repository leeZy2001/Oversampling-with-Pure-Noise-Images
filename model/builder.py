from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model

from model.darbn import CustomDAR_BN

def build_model(input_shape, num_classes, model_config):
    inputs = layers.Input(shape=input_shape)
    last_layer = inputs
    for layer_config in model_config["layers"]:
        last_layer = build_layer(*layer_config)(last_layer)
    outputs = Dense(num_classes, *model_config["output"])(last_layer)
    model = Model(name=model_config["name"], inputs=inputs, outputs=outputs)
    return model


def build_optimizer(model_config):
    match model_config["optimizer"]:
        case 'adam':
            return optimizers.Adam()
    return None


def build_layer(layer_type, **options):
    match layer_type:
        case 'conv2d':
            return layers.Conv2D(*options)
        case 'relu':
            return layers.ReLU(*options)
        case 'darbn':
            return CustomDAR_BN(*options)
        case 'flatten':
            return layers.Flatten(*options)
    return None

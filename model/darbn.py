import tensorflow as tf
from tensorflow.keras.layers import Layer


def lerp(from: float, to: float, t: float) -> float:
    return from * t + to * (1 - t)


class CustomDAR_BN(Layer):
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-3):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=(input_shape[-1],),
                                           initializer='zeros'
                                           trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=(input_shape[-1],),
                                               initializer='ones',
                                               trainable=False)
    def _batch_norm(self, x, mean, var):
        return self.gamma * (x - mean) / (tf.sqrt(var + self.epsilon)) + self.beta

    def call(self, inputs, training=None):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs,
                                                       axes=[0,1,2],
                                                       keepdims=False)
            self.moving_mean.assign(lerp(self.moving_mean,
                                         batch_mean,
                                         self.momentum))
            self.moving_variance.assign(lerp(self.moving_variance,
                                             batch_variance,
                                             self.momentum))
            return self._batch_norm(inputs, batch_mean, batch_variance)
        return self._batch_norm(inputs,
                                self.moving_mean,
                                self.moving_variance)


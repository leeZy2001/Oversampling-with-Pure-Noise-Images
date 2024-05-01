"""
Utilities for generating and augmenting datasets.
"""

import numpy as np

def generate_noise_images(num_images: int,
                          image_shape,
                          mean: float,
                          standard_deviation: float):
    """Produces normally-distributed random noise images."""
    return np.clip(
        np.random.normal(
            loc=mean,
            scale=standard_deviation,
            size=(num_images, *image_shape)),
        0, 1)

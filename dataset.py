import numpy as np

def generate_noise_images(num_images: int,
                          image_shape,
                          mean: float,
                          standard_deviation: float):
    return np.clip(
        np.random.normal(
            loc=mean,
            scale=standard_deviation,
            size=(num_images, *image_shape)),
        0, 1)


"""
Scripts that, having a Keras model and some data about its input,
calculates a lower limit for the memory usage
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import numpy as np
import argparse


def load_model(model_path):
    ''' Loads and returns a keras model '''
    return keras.models.load_model(model_path)


def find_dtype(model):
    ''' Returns the data type of the parameters / weights / activations of the model '''
    return model.inputs[0].dtype


def find_lower_limit(model, dtype=None):
    """
    Calculate a lower limit for the memory usage of the model

    Arguments:
        model: a Keras model
        dtype: the data type of the model's parameter / weights / activations ...

    Returns: the lower limit of the model's memory usage
    """

    # If no data type is provided, use the function to get it
    if dtype == None:
        dtype = find_dtype(model)

    # Go through the model and find the lower limit of its memory usage
    limit = 0
    import numpy as np
    for i in range(len(model.layers)):
        layer = model.layers[i]
        type = layer.__class__.__name__
        layer_params = 0
        if type == 'Conv2D':
            shape = np.shape(layer.get_weights()[0])
            layer_params = shape[0]
            layer_params = layer_params * shape[i] for i in range(1, len(shape))
            if len(layer.get_weights()) > 1:
                layer_params = layer_params + np.shape(layer.get_weights()[1])[0]

        elif type == 'PReLU':
            shape = np.shape(layer.get_weights())
            layer_params = shape[0]
            layer_params = layer_params * shape[i] for i in range(1, len(shape))

        limit = limit + layer_params * dtype


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Having a Keras model and some data about its input, calculates a lower limit for the memory usage"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path) of the .h5 file containing the Keras model")
    args = parser.parse_args()

    model = load_model(args.keras_model)
    data_type = find_dtype(model)

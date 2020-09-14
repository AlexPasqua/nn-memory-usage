"""
Scripts that, having a Keras model and some data about its input,
calculates a lower limit for the memory usage
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import numpy as np
import argparse
import re


KILO = 1000.0
MILLION = 1000000.0
BILLION = 1000000000.0

def my_regex_type(arg_value, pat=re.compile(r"^[0-9]+x[0-9]+x3{0,1}$")):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError
    return arg_value


def load_model(model_path):
    ''' Loads and returns a keras model '''
    return keras.models.load_model(model_path)


def find_dtype(model):
    ''' Returns the data type of the parameters / weights / activations of the model '''
    return model.inputs[0].dtype


def express_dtype_in_bytes(dtype_str):
    if dtype_str == 'float32': return 4
    elif dtype_str == 'float16': return 2
    elif dtype_str == 'float64': return 8
    else: print('E: data type {} not supported'.format(dtype_str))


def calc_params_Conv2D(layer):
    """
    Calculates the number of parameters for a Conv2D Keras layer
    Arguments:
        layer: a Keras layer (Conv2D)
    Returns: the number of this layer's paramenters
    """
    shape = np.shape(layer.get_weights()[0])
    layer_params = shape[0]
    for i in range(1, len(shape)):
        layer_params = layer_params * shape[i]
    if len(layer.get_weights()) > 1:
        layer_params = layer_params + np.shape(layer.get_weights()[1])[0]
    return layer_params


def calc_params_PReLU(layer):
    """
    Calculates the number of parameters for a PReLU Keras layer
    Arguments:
        layer: a Keras layer (PReLU)
    Returns: the number of this layer's paramenters
    """
    shape = np.shape(layer.get_weights())
    layer_params = shape[0]
    for i in range(1, len(shape)):
        layer_params = layer_params * shape[i]
    return layer_params


def calc_params_Dense(layer):
    """
    Calculates the number of parameters for a Dense Keras layer
    Arguments:
        layer: a Keras layer (Dense)
    Returns: the number of this layer's paramenters
    """
    layer_params = 0
    weights = layer.get_weights()[0]
    layer_params = layer_params + np.shape(weights)[0] * len(weights[0])
    if len(layer.get_weights()) > 1:
        layer_params = layer_params + len(layer.get_weights()[1])
    return layer_params


def calc_input_size(n_inputs, model, dtype=None, input_size=None):
    """
    Calculates the memory used by the inputs given to the NN

    Arguments:
        n_inputs: the number of input images given to the NN
        model: the keras model
        dtype: (optional) the data type of the input.
                If missing, the dtype of the model will be used if model_path != None, otherwise
                we'll assume float32 as input data type
        input_size: (optional) the input_size. If missing we'll take the model's default one

    Returns: the memory used by the inputs in Bytes
    """

    if dtype == None:
        dtype = find_dtype(model)

    if input_size == None:
        input_size = model.inputs[0].shape[1 : ]
        height = input_size[0]
        width = input_size[1]
        channels = input_size[2]
    else:
        input_size = input_size.split('x')
        height = int(input_size[0])
        width = int(input_size[1])
        if len(input_size) > 2:
            channels = int(input_size[2])
        else:
            channels = 1

    return height * width * channels * n_inputs * express_dtype_in_bytes(dtype)


def find_lower_limit(argv):
    """
    Calculates a lower limit for the memory usage of the model

    Arguments:
        model: a Keras model
        dtype: the data type of the model's parameter / weights / activations ...

    Returns: the lower limit of the model's memory usage in Bytes
    """

    verbose = argv.verbose

    # Load model
    model = load_model(argv.keras_model)

    # Retrieve the model's data type
    dtype = find_dtype(model)
    if verbose:
        print(f'Data type: {dtype}')
    # dtype will be expressed in bytes
    dtype = express_dtype_in_bytes(dtype)

    # Go through the model and find the lower limit of its memory usage
    limit = 0
    import numpy as np
    for i in range(len(model.layers)):
        layer = model.layers[i]
        type = layer.__class__.__name__
        layer_params = 0

        if verbose:
            print(f'\nLayer: {layer.name}\nType: {type}')

        if type in ('Conv2D', 'PReLU', 'Dense'):
            # Dictionary where a type corresponds to a function name
            functions_map = {'Conv2D': calc_params_Conv2D, 'PReLU': calc_params_PReLU, 'Dense': calc_params_Dense}
            layer_params = functions_map[type](layer)   # call the function whoes name is the value of functions_map[type]

        limit = limit + layer_params * dtype
        if verbose:
            print(f"This layer's number of parameters: {layer_params}")
            print(f'Current lower limit of memory usage: {limit} Bytes')

    return limit + calc_input_size(argv.n_inputs, model, argv.input_dtype, argv.input_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Having a Keras model and some data about its input, calculates a lower limit for the memory usage"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path) of the .h5 file containing the Keras model")
    parser.add_argument('n_inputs', action='store', type=int, help="The number of input images to be processed 'at the same time'")
    parser.add_argument('--input_size', action='store', type=my_regex_type, help="Input size as <height>x<width>x3 if RGB, if grayscale <height>x<width>")
    parser.add_argument('--input_dtype', action='store', help="The input's data type")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    limit = find_lower_limit(args)

    # Find best dividend to represent the result
    if limit >= BILLION:
        dividend = BILLION
        unit_of_measure = 'GB'
    elif limit >= MILLION:
        dividend = MILLION
        unit_of_measure = 'MB'
    elif limit >= KILO:
        dividend = KILO
        unit_of_measure = 'KB'
    else:
        dividend = 1.0
        unit_of_measure = 'Bytes'

    print('\nMemory usage (model + inputs) lower limit: {} {}\n'.format(float(limit) / dividend, unit_of_measure))

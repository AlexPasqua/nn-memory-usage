# NN memory usage lower limit finder
Given a Keras model and the shape of the input data, ```limit_finder.py``` will calculate a lower limit for the memory usage of the model.

### Usage
```
python3 limit_finder.py <keras_model> [--n_inputs N_INPUTS] [--input_size INPUT_SIZE] [--input_dtype INPUT_DTYPE] [--verbose]
```
Where:
* ```keras_model``` is the path to a Keras model
* ```N_INPUTS``` is an integer indicating the number of images to be processed 'at the same time'
* ```INPUT_SIZE``` is a string matching ```<height>x<width>x3``` if RGB, ```<height>x<width>``` if grayscale. It's actually a regular expression matching ```^[0-9]+x[0-9]+x3{0,1}$```
* ```INPUT_DTYPE``` is a string among {float32 | float16 | float64}

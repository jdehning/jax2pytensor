# Conv-Jax-Pytensor

This tool transform [JAX](https://jax.readthedocs.io/en/latest/) functions into 
[PyTensor](https://pytensor.readthedocs.io/en/latest/) compatible functions.
Based on the 

* Documentation: https://conv-jax-pytensor.readthedocs.io.

## Features

* Backward differenciation of the transformed function works.
* Arguments and return values can be arbitrary nested python structures (pytrees).
* Automatic inference of the shape and structure of return values, needed for PyTensor.





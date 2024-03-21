# jax2pytensor

This tool transforms [JAX](https://jax.readthedocs.io/en/latest/) functions into 
[PyTensor](https://pytensor.readthedocs.io/en/latest/) compatible functions.
Idea originates from the blog entry from Ricardo Vieira and Adrian Seyboldt 
[here](https://www.pymc-labs.com/blog-posts/jax-functions-in-pymc-3-quick-examples/).
Useful for using JAX functions in [PyMC](https://www.pymc.io/welcome.html) models.

* Documentation: https://jax2pytensor.readthedocs.io.

## Features

* Backward differentiation of the transformed function works.
* Arguments and return values can be arbitrary nested python structures ([pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)).
* Automatic inference of the shape and structure of return values, needed for the internal graph construction of PyTensor.

## Limits

* Forward differentiation of the transformed function is not implemented.
* All return values must be transformable into pytensor variables. 
* Unknown input shapes at graph building time might not work (as returned by a pytensor scan/map function)





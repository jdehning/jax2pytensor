# Bayesian ODE

This toolbox aims to simplify the construction ODEs and the inference of their parameters.

The aim isn't to provide a complete package that will build models from A to Z, but rather
provide different helper functions examples and guidelines to help leverage modern python
packages like [JAX](https://jax.readthedocs.io/en/latest/),
[Diffrax](https://docs.kidger.site/diffrax/) and
[PyMC](https://www.pymc.io/welcome.html) to build, automatically differentiate and fit
compartmental models.

* Documentation: https://bayesian_ode.readthedocs.io.

## Features

* Integrate the ODEs using diffrax, automatically generating the Jacobian of the parameters of the ODE
* Fit the parameters using minimization algorithms or build a Bayesian model using PyMC.





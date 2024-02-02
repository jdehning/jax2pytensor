# mypy: disable-error-code="attr-defined"
# isort: skip_file
"""Inference of Compartmental Models Toolbox."""
from importlib import metadata as importlib_metadata

from ._comp_model import (
    ODEIntegrator,
    interpolate_pytensor,
    interpolation_func,
)

from ._pytensor_op import create_and_register_jax

from ._slow_modulation import priors_for_cps, sigmoidal_changepoints
from ._tools import hierarchical_priors


def _get_version():
    try:
        return importlib_metadata.version("bayesian_ODE")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__ = _get_version()

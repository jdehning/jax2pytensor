# mypy: disable-error-code="attr-defined"
# isort: skip_file
"""Inference of Compartmental Models Toolbox."""
from importlib import metadata as importlib_metadata


from ._jax_to_pytensor import jax_to_pytensor
from ._diffeq import diffeqsolve


def _get_version():
    try:
        return importlib_metadata.version("conv_jax_pytensor")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__ = _get_version()

# mypy: disable-error-code="attr-defined"
# isort: skip_file
"""Inference of Compartmental Models Toolbox."""
from importlib import metadata as importlib_metadata


from ._jax2pytensor import jax2pytensor
from ._diffeq import diffeqsolve


def _get_version():
    try:
        return importlib_metadata.version("jax2pytensor")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__ = _get_version()

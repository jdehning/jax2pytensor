import diffrax
import jax

from ._jax2pytensor import jax2pytensor


def diffeqsolve(*other_args, **other_kwargs):
    """
    Solve the ODE system with initial conditions y0 and parameters params
    """
    func = jax2pytensor(
        diffrax.diffeqsolve,
        args_for_graph=["y0", "args"],
        output_formatter=lambda solution: solution.ys,
    )
    return func(*other_args, **other_kwargs)

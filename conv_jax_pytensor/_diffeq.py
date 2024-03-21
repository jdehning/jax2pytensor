import diffrax
import jax

from ._jax_to_pytensor import jax_to_pytensor


def diffeqsolve(*other_args, **other_kwargs):
    """
    Solve the ODE system with initial conditions y0 and parameters params
    """
    func = jax_to_pytensor(
        diffrax.diffeqsolve,
        args_for_graph=["y0", "args"],
        output_formatter=lambda solution: solution.ys,
    )
    return func(*other_args, **other_kwargs)

from functools import wraps

import diffrax
import jax


from ._jax2pytensor import jax2pytensor


def diffeqsolve(*other_args, **other_kwargs):
    """
    Solve the ODE system with initial conditions y0 and parameters params
    """

    # Only return the the solution: ys
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs).ys

        return wrapped

    func = jax2pytensor(
        wrapper(diffrax.diffeqsolve),
        args_for_graph=["y0", "args"],
    )
    return func(*other_args, **other_kwargs)

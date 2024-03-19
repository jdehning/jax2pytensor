import diffrax
import jax

from ._pytensor_op import jaxfunc_to_pytensor


def diffeqsolve(*other_args, **other_kwargs):
    """
    Solve the ODE system with initial conditions y0 and parameters params
    """
    add_dim = lambda x: x.shape + (None,)

    func = jaxfunc_to_pytensor(
        diffrax.diffeqsolve,
        args_for_graph=["y0", "args"],
        output_shape_def=lambda y0, args: jax.tree_util.tree_map(add_dim, y0),
        output_formatter=lambda solution: solution.ys,
    )
    return func(*other_args, **other_kwargs)
